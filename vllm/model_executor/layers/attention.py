"""Multi-head attention."""

from typing import List, Optional

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalMask,
    LowerTriangularMaskWithTensorBias,
)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip
from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    is_fa_version_supported,
)

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(
                f"head_size ({self.head_size}) is not supported. "
                f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}."
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
            )

        if input_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                # project the key and value tensors to the desired number of
                # heads.
                # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                query = query.view(
                    query.shape[0],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    query.shape[-1],
                )
                key = key[:, :, None, :].expand(
                    key.shape[0],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    key.shape[-1],
                )
                value = value[:, :, None, :].expand(
                    value.shape[0],
                    self.num_kv_heads,
                    self.num_queries_per_kv,
                    value.shape[-1],
                )

            # Set attention bias if not provided. This typically happens at the
            # very attention layer of every iteration.
            # FIXME(woosuk): This is a hack.
            if input_metadata.attn_bias is None:
                if self.alibi_slopes is None:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        [seq_len] * batch_size
                    )
                    if self.sliding_window is not None:
                        attn_bias = attn_bias.make_local_attention(self.sliding_window)
                    input_metadata.attn_bias = attn_bias
                else:
                    input_metadata.attn_bias = _make_alibi_bias(
                        self.alibi_slopes,
                        self.num_kv_heads,
                        batch_size,
                        seq_len,
                        query.dtype,
                    )

            # Hydragen-style prefill path if shared segments are provided.
            if (
                getattr(input_metadata, "shared_ks", None) is not None
                and getattr(input_metadata, "shared_vs", None) is not None
                and len(input_metadata.shared_ks) > 0
                and len(input_metadata.shared_vs) > 0
            ):
                # Only single shared segment supported in v1 for simplicity.
                shared_k = input_metadata.shared_ks[0]
                shared_v = input_metadata.shared_vs[0]

                # Validate shapes: Ks, Vs: [Bs, S, num_kv_heads, head_dim]
                assert shared_k.dim() == 4 and shared_v.dim() == 4, (
                    f"shared_k/shared_v must be 4D, got {shared_k.shape} / {shared_v.shape}"
                )
                assert shared_k.shape == shared_v.shape, (
                    f"shared_k/shared_v shape mismatch: {shared_k.shape} vs {shared_v.shape}"
                )
                Bs, S, kv_heads_s, hd_s = shared_k.shape
                assert kv_heads_s == self.num_kv_heads and hd_s == self.head_size, (
                    f"Shared KV heads/hdim mismatch: {(kv_heads_s, hd_s)} vs {(self.num_kv_heads, self.head_size)}"
                )
                assert batch_size % Bs == 0, (
                    f"Batch {batch_size} must be divisible by shared batch {Bs}"
                )
                groups = batch_size // Bs

                # Reconstruct Q, K, V for unique path as 4D tensors [B, Q, H, D].
                # Query may be [B*Q, H] or [B*Q, kvH, qPerKV, D]; reshape to [B*Q, H, D] then to [B, Q, H, D].
                q_flat = query.reshape(-1, self.num_heads, self.head_size)
                q4 = q_flat.view(batch_size, seq_len, self.num_heads, self.head_size)

                # Key/Value may have been expanded for MQA/GQA: drop the repeated query-per-kv axis if present.
                if key.dim() == 4:
                    key_kv = key[:, :, 0, :]
                    value_kv = value[:, :, 0, :]
                else:
                    key_kv = key
                    value_kv = value
                k4 = key_kv.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
                v4 = value_kv.view(
                    batch_size, seq_len, self.num_kv_heads, self.head_size
                )

                # Build varlen inputs for unique segment (per batch sequence of length seq_len).
                total_q = batch_size * seq_len
                cu_q = torch.arange(
                    0,
                    (batch_size + 1) * seq_len,
                    step=seq_len,
                    dtype=torch.int32,
                    device=q4.device,
                )
                q_unique = q4.reshape(total_q, self.num_heads, self.head_size)
                k_unique = k4.reshape(
                    batch_size * seq_len, self.num_kv_heads, self.head_size
                )
                v_unique = v4.reshape(
                    batch_size * seq_len, self.num_kv_heads, self.head_size
                )

                cu_k_unique = cu_q  # same layout (each sequence has length seq_len)

                # Compute unique attention with LSE (causal=True).
                # Choose FlashAttention version if available; else fall back.
                fa_ver = (
                    3
                    if is_fa_version_supported(3, q4.device)
                    else (2 if is_fa_version_supported(2, q4.device) else 0)
                )
                if fa_ver:
                    out_u, lse_u = flash_attn_varlen_func(
                        q_unique,
                        k_unique,
                        v_unique,
                        max_seqlen_q=seq_len,
                        cu_seqlens_q=cu_q,
                        max_seqlen_k=seq_len,
                        cu_seqlens_k=cu_k_unique,
                        dropout_p=0.0,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        return_softmax_lse=True,
                        fa_version=fa_ver,
                    )
                else:
                    # Fallback: compute concatenated attention directly in torch (slow path)
                    ks_b = shared_k.repeat(batch_size // Bs, 1, 1, 1)
                    vs_b = shared_v.repeat(batch_size // Bs, 1, 1, 1)
                    kcat = torch.cat([ks_b, k4], dim=1)
                    vcat = torch.cat([vs_b, v4], dim=1)
                    scores = (
                        self.scale * torch.einsum("bqhd,bkhd->bhqk", q4, kcat).float()
                    )
                    mask_shared = torch.zeros(
                        seq_len, S, device=q4.device, dtype=scores.dtype
                    )
                    tri = torch.triu(
                        torch.ones(
                            seq_len, seq_len, device=q4.device, dtype=scores.dtype
                        ),
                        diagonal=1,
                    )
                    mask_unique = tri * torch.finfo(scores.dtype).min
                    mask = torch.cat([mask_shared, mask_unique], dim=1)
                    scores = scores + mask.unsqueeze(0).unsqueeze(0)
                    probs = torch.softmax(scores, dim=-1).to(vcat.dtype)
                    out_cat = torch.einsum("bhqk,bkhd->bqhd", probs, vcat)
                    output = out_cat.reshape(
                        batch_size * seq_len, self.num_heads, self.head_size
                    )
                    return output.view(batch_size, seq_len, hidden_size)
                # Reshape to [B, Q, H, D] and [B, Q, H].
                out_u = out_u.view(batch_size, seq_len, self.num_heads, self.head_size)
                # lse_u: (H, total_q) -> [B, Q, H]
                lse_u = (
                    lse_u.view(self.num_heads, batch_size, seq_len)
                    .permute(1, 2, 0)
                    .contiguous()
                )

                # Build varlen inputs for shared segment.
                # q grouped to [Bs, groups*Q, H, D]
                q_grouped = q4.view(Bs, groups, seq_len, self.num_heads, self.head_size)
                q_grouped = q_grouped.reshape(
                    Bs, groups * seq_len, self.num_heads, self.head_size
                )
                total_q_shared = Bs * groups * seq_len
                q_shared = q_grouped.reshape(
                    total_q_shared, self.num_heads, self.head_size
                )
                cu_q_shared = torch.arange(
                    0,
                    (Bs + 1) * (groups * seq_len),
                    step=(groups * seq_len),
                    dtype=torch.int32,
                    device=q4.device,
                )

                # shared_k/v flattened: [Bs*S, kvH, D]
                k_shared = shared_k.reshape(Bs * S, self.num_kv_heads, self.head_size)
                v_shared = shared_v.reshape(Bs * S, self.num_kv_heads, self.head_size)
                cu_k_shared = torch.arange(
                    0, (Bs + 1) * S, step=S, dtype=torch.int32, device=q4.device
                )

                if fa_ver:
                    out_s, lse_s = flash_attn_varlen_func(
                        q_shared,
                        k_shared,
                        v_shared,
                        max_seqlen_q=groups * seq_len,
                        cu_seqlens_q=cu_q_shared,
                        max_seqlen_k=S,
                        cu_seqlens_k=cu_k_shared,
                        dropout_p=0.0,
                        softmax_scale=self.scale,
                        causal=False,
                        alibi_slopes=None,
                        return_softmax_lse=True,
                        fa_version=fa_ver,
                    )
                else:
                    # handled by earlier fallback return
                    pass
                # Reshape to [B, Q, H, D] and [B, Q, H].
                out_s = out_s.view(Bs, groups, seq_len, self.num_heads, self.head_size)
                out_s = out_s.reshape(
                    batch_size, seq_len, self.num_heads, self.head_size
                )
                lse_s = (
                    lse_s.view(self.num_heads, Bs, groups * seq_len)
                    .permute(1, 2, 0)
                    .contiguous()
                )
                lse_s = lse_s.view(batch_size, seq_len, self.num_heads)

                # Two-way LSE combine per (B, Q, H, D).
                output_merged = _combine_lse_two_way(out_s, lse_s, out_u, lse_u)
                output = output_merged.view(-1, self.num_heads, self.head_size)
            else:
                # TODO(woosuk): Too many view operations. Let's try to reduce them
                # in the future for code readability.
                if self.alibi_slopes is None:
                    query = query.unsqueeze(0)
                    key = key.unsqueeze(0)
                    value = value.unsqueeze(0)
                else:
                    query = query.unflatten(0, (batch_size, seq_len))
                    key = key.unflatten(0, (batch_size, seq_len))
                    value = value.unflatten(0, (batch_size, seq_len))

                out = xops.memory_efficient_attention_forward(
                    query,
                    key,
                    value,
                    attn_bias=input_metadata.attn_bias,
                    p=0.0,
                    scale=self.scale,
                    op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0]
                    if (is_hip())
                    else None,
                )
                output = out.view_as(query)
        else:
            # Decoding run.
            if key_cache is not None and value_cache is not None:
                output = _paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                )
            else:
                # This happens during the initial memory profiling run for
                # CUDA graphs.
                output = torch.zeros_like(query)

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype, device="cuda")
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _combine_lse_two_way(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Two-way LSE combine for tensors shaped [B, Q, H, D] and [B, Q, H].

    Computes: (out_a * exp(lse_a - m) + out_b * exp(lse_b - m)) / (exp(lse_a - m) + exp(lse_b - m)),
    where m = max(lse_a, lse_b) computed elementwise over [B, Q, H].
    """
    # Promote to fp32 for stability in exponentials and division.
    lse_a_32 = lse_a.float()
    lse_b_32 = lse_b.float()
    m = torch.maximum(lse_a_32, lse_b_32)
    adj_a = torch.exp(lse_a_32 - m)
    adj_b = torch.exp(lse_b_32 - m)
    denom = adj_a + adj_b
    # Avoid division by zero (degenerate case), should not happen with valid LSEs.
    denom = torch.maximum(denom, torch.finfo(denom.dtype).tiny)
    out_a_32 = out_a.float()
    out_b_32 = out_b.float()
    combined = (
        out_a_32 * adj_a.unsqueeze(-1) + out_b_32 * adj_b.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    return combined.to(out_a.dtype)


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(query)

    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
        input_metadata.max_context_len + _PARTITION_SIZE - 1
    ) // _PARTITION_SIZE
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = input_metadata.max_context_len <= 8192 and (
        max_num_partitions == 1 or num_seqs * num_heads > 512
    )
    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    return output
