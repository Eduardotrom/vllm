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

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

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

    def _prompt_with_shared_prefix(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Hydragen-style prefill using a single shared prefix.

        Returns flattened output of shape [B*Q, H, D].
        """
        # Shared prefix length and groups
        S = int(input_metadata.shared_prefix_len)
        Bs = int(getattr(input_metadata, "shared_groups", 1) or 1)
        assert S <= seq_len, f"shared_prefix_len {S} > seq_len {seq_len}"
        assert batch_size % Bs == 0, (
            f"Batch {batch_size} must be divisible by shared_groups {Bs}"
        )
        groups = batch_size // Bs

        # Reconstruct 4D tensors: q4=[B,Q,H,D], k4/v4=[B,Q,Hkv,D]
        q_flat = query.reshape(-1, self.num_heads, self.head_size)
        q4 = q_flat.view(batch_size, seq_len, self.num_heads, self.head_size)
        if key.dim() == 4:
            key_kv = key[:, :, 0, :]
            value_kv = value[:, :, 0, :]
        else:
            key_kv = key
            value_kv = value
        k4 = key_kv.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        v4 = value_kv.view(batch_size, seq_len, self.num_kv_heads, self.head_size)

        # Prepare shared K/V buffers (from cache if available), reused below
        cached_k = getattr(input_metadata, "shared_prefix_k", None)
        cached_v = getattr(input_metadata, "shared_prefix_v", None)
        cached_cu = getattr(input_metadata, "shared_prefix_cu", None)
        if cached_k is not None and cached_v is not None and cached_cu is not None:
            shared_k = cached_k.view(Bs, S, self.num_kv_heads, self.head_size)
            shared_v = cached_v.view(Bs, S, self.num_kv_heads, self.head_size)
            k_shared = cached_k  # [Bs*S, Hkv, D]
            v_shared = cached_v
            cu_k_shared = cached_cu  # [Bs+1]
        else:
            shared_k_list: List[torch.Tensor] = []
            shared_v_list: List[torch.Tensor] = []
            for g in range(Bs):
                base_idx = g * groups
                shared_k_list.append(k4[base_idx, :S])
                shared_v_list.append(v4[base_idx, :S])
            shared_k = torch.stack(shared_k_list, dim=0)  # [Bs, S, Hkv, D]
            shared_v = torch.stack(shared_v_list, dim=0)
            k_shared = shared_k.reshape(Bs * S, self.num_kv_heads, self.head_size)
            v_shared = shared_v.reshape(Bs * S, self.num_kv_heads, self.head_size)
            cu_k_shared = torch.arange(
                0, (Bs + 1) * S, step=S, dtype=torch.int32, device=q4.device
            )
        Lu = seq_len - S
        # Group queries: [Bs, groups*Q, H, D]
        q_grouped = q4.view(Bs, groups, seq_len, self.num_heads, self.head_size)
        q_grouped = q_grouped.reshape(
            Bs, groups * seq_len, self.num_heads, self.head_size
        )
        total_q_shared = Bs * groups * seq_len
        q_shared = q_grouped.reshape(total_q_shared, self.num_heads, self.head_size)
        cu_q_shared = torch.arange(
            0,
            (Bs + 1) * (groups * seq_len),
            step=(groups * seq_len),
            dtype=torch.int32,
            device=q4.device,
        )

        fa_ver = (
            3
            if is_fa_version_supported(3, q4.device)
            else (2 if is_fa_version_supported(2, q4.device) else 0)
        )
        if not fa_ver:
            raise ValueError("Not compatible attention function")
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
            return_softmax_lse=False,
            fa_version=fa_ver,
        )
        out_s = out_s.view(Bs, groups, seq_len, self.num_heads, self.head_size)
        out_s = out_s.reshape(batch_size, seq_len, self.num_heads, self.head_size)

        # If only the shared part exists, compute shared attention only (non-causal)
        if Lu == 0:
            # shared_k, k_shared, v_shared, cu_k_shared prepared above
            return out_s.reshape(-1, self.num_heads, self.head_size)
        # Compute two-way combine (shared non-causal, unique causal)
        total_q = batch_size * seq_len
        cu_q = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=q4.device,
        )
        q_unique = q4.reshape(total_q, self.num_heads, self.head_size)
        k_unique = k4[:, S:, :, :].reshape(
            batch_size * Lu, self.num_kv_heads, self.head_size
        )
        v_unique = v4[:, S:, :, :].reshape(
            batch_size * Lu, self.num_kv_heads, self.head_size
        )
        cu_k_unique = torch.arange(
            0, (batch_size + 1) * Lu, step=Lu, dtype=torch.int32, device=q4.device
        )

        out_u, lse_u = flash_attn_varlen_func(
            q_unique,
            k_unique,
            v_unique,
            max_seqlen_q=seq_len,
            cu_seqlens_q=cu_q,
            max_seqlen_k=Lu,
            cu_seqlens_k=cu_k_unique,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            return_softmax_lse=True,
            fa_version=fa_ver,
        )

        # Reshape unique outputs
        out_u = out_u.view(batch_size, seq_len, self.num_heads, self.head_size)
        lse_u = (
            lse_u.view(self.num_heads, batch_size, seq_len)
            .permute(1, 2, 0)
            .contiguous()
        )

        lse_s = (
            lse_s.view(self.num_heads, Bs, groups * seq_len)
            .permute(1, 2, 0)
            .contiguous()
        )
        lse_s = lse_s.view(batch_size, seq_len, self.num_heads)

        output_merged = _combine_lse_many([out_s, out_u], [lse_s, lse_u])
        return output_merged.view(-1, self.num_heads, self.head_size)

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

            # Hydragen-style prefill with shared-prefix only (single shared segment).
            if (
                getattr(input_metadata, "shared_prefix_len", None) is not None
                and int(input_metadata.shared_prefix_len) > 0
            ):
                output = self._prompt_with_shared_prefix(
                    query, key, value, input_metadata, batch_size, seq_len
                )
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
            fa_ver = (
                3
                if is_fa_version_supported(3, query.device)
                else (2 if is_fa_version_supported(2, query.device) else 0)
            )
            use_hydragen_decode = (
                getattr(input_metadata, "use_hydragen_decode", False)
                and getattr(input_metadata, "shared_prefix_len", None) is not None
                and int(input_metadata.shared_prefix_len) > 0
                and fa_ver
            )

            if (
                use_hydragen_decode
                and key_cache is not None
                and value_cache is not None
            ):
                output = self._decode_with_shared_prefix(
                    query, key_cache, value_cache, input_metadata
                )
            elif key_cache is not None and value_cache is not None:
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

    def _decode_with_shared_prefix(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Hydragen-style decode using FA-varlen over shared prefix and unique past.

        Assumes one query token per sequence (standard decode). Packs K/V from
        the paged cache into contiguous buffers via gather, runs varlen FA for
        shared (non-causal) and unique (causal) segments, then LSE-combines.

        Limitations:
        - Single shared segment derived from the first sequence of each group.
        - Requires CUDA FlashAttention backend (FA2/FA3) availability.
        - bf16 recommended; falls back to the dtype of inputs.
        """
        device = query.device
        dtype = query.dtype
        batch_size, num_heads, head_size = query.shape

        # Determine grouping and shared prefix length
        Bs = int(getattr(input_metadata, "shared_groups", 1) or 1)
        assert batch_size % Bs == 0, (
            f"Batch {batch_size} must be divisible by shared_groups {Bs}"
        )
        groups = batch_size // Bs
        S = int(input_metadata.shared_prefix_len)

        # Build per-sequence context lengths
        # context_lens: [B]
        context_lens = input_metadata.context_lens.to(device=device)
        # Block size from cache layout
        block_size = int(value_cache.shape[3])

        # Build slot mapping for full unique contexts across the batch
        # slot_mapping_unique: [sum_i L_i] with slot = block_id * block_size + offset
        block_tables = input_metadata.block_tables.to(device=device)
        slots_list = []
        cu_k_unique = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
        cu = 0
        cu_k_unique[0] = 0
        for i in range(batch_size):
            L = int(context_lens[i].item())
            if L < 0:
                L = 0
            pos = torch.arange(L, device=device, dtype=torch.int32)
            block_idx = torch.div(pos, block_size, rounding_mode="floor")
            num_blocks = int((L + block_size - 1) // block_size)
            blocks_i = block_tables[i, :num_blocks].to(torch.int32)
            block_ids = blocks_i.index_select(0, block_idx)
            offsets = torch.remainder(pos, block_size)
            slots = block_ids * block_size + offsets
            slots_list.append(slots)
            cu += L
            cu_k_unique[i + 1] = cu

        total_k = int(cu)
        if total_k == 0:
            # Degenerate: no history, output zeros
            return torch.zeros_like(query)

        # Gather unique K/V into contiguous buffers: [total_k, Hkv, D]
        k_unique = torch.empty(
            (total_k, self.num_kv_heads, head_size), device=device, dtype=dtype
        )
        v_unique = torch.empty_like(k_unique)
        slot_mapping_unique = torch.cat(slots_list, dim=0).contiguous()
        # cache ops expects int32 slot mapping
        cache_ops.gather_cached_kv(
            k_unique, v_unique, key_cache, value_cache, slot_mapping_unique
        )

        # Prepare queries for varlen FA
        q4 = query.view(batch_size, 1, num_heads, head_size)

        # Build shared K/V from the first sequence in each group: [Bs, S, Hkv, D]
        # Validate S does not exceed context of base sequences
        for g in range(Bs):
            base_idx = g * groups
            # CURSOR: Review this line - Assuming shared prefix length S <= context_lens[base_idx]
            assert S <= int(context_lens[base_idx].item()), (
                f"shared_prefix_len {S} exceeds context length of base seq {base_idx}"
            )

        # Build slot mapping for shared prefixes (concatenate Bs segments)
        slots_shared_list = []
        for g in range(Bs):
            base_idx = g * groups
            Ls = S
            pos = torch.arange(Ls, device=device, dtype=torch.int32)
            block_idx = torch.div(pos, block_size, rounding_mode="floor")
            num_blocks = int((Ls + block_size - 1) // block_size)
            blocks_base = block_tables[base_idx, :num_blocks].to(torch.int32)
            block_ids = blocks_base.index_select(0, block_idx)
            offsets = torch.remainder(pos, block_size)
            slots = block_ids * block_size + offsets
            slots_shared_list.append(slots)
        slot_mapping_shared = torch.cat(slots_shared_list, dim=0).contiguous()

        k_shared = torch.empty(
            (Bs * S, self.num_kv_heads, head_size), device=device, dtype=dtype
        )
        v_shared = torch.empty_like(k_shared)
        cache_ops.gather_cached_kv(
            k_shared, v_shared, key_cache, value_cache, slot_mapping_shared
        )
        cu_k_shared = torch.arange(
            0, (Bs + 1) * S, step=S, dtype=torch.int32, device=device
        )

        # Group queries for shared segment: [Bs, groups*1, H, D] -> [total_q_shared, H, D]
        q_grouped = q4.view(Bs, groups, 1, num_heads, head_size)
        q_grouped = q_grouped.reshape(Bs, groups * 1, num_heads, head_size)
        total_q_shared = Bs * groups * 1
        q_shared = q_grouped.reshape(total_q_shared, num_heads, head_size)
        cu_q_shared = torch.arange(
            0,
            (Bs + 1) * (groups * 1),
            step=(groups * 1),
            dtype=torch.int32,
            device=device,
        )

        # Build unique varlen descriptors
        q_unique = query.reshape(batch_size, num_heads, head_size)
        cu_q_unique = torch.arange(
            0, batch_size + 1, step=1, dtype=torch.int32, device=device
        )

        # Select FA version
        fa_ver = (
            3
            if is_fa_version_supported(3, device)
            else (2 if is_fa_version_supported(2, device) else 0)
        )
        # Fa version availability is validated before this function so
        # at this point we can assume fa_ver is valid

        # Run shared (non-causal) attention
        out_s, lse_s = flash_attn_varlen_func(
            q_shared,
            k_shared,
            v_shared,
            max_seqlen_q=groups * 1,
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
        out_s = out_s.view(Bs, groups, 1, num_heads, head_size)
        out_s = out_s.reshape(batch_size, 1, num_heads, head_size)
        lse_s = lse_s.view(num_heads, Bs, groups * 1).permute(1, 2, 0).contiguous()
        lse_s = lse_s.view(batch_size, 1, num_heads)

        # Run unique (causal) attention
        out_u, lse_u = flash_attn_varlen_func(
            q_unique,
            k_unique,
            v_unique,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q_unique,
            max_seqlen_k=int(context_lens.max().item()),
            cu_seqlens_k=cu_k_unique,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            return_softmax_lse=True,
            fa_version=fa_ver,
        )
        out_u = out_u.view(batch_size, 1, num_heads, head_size)
        lse_u = lse_u.view(num_heads, batch_size, 1).permute(1, 2, 0).contiguous()

        # Combine and return
        out = _combine_lse_two_way(out_s, lse_s, out_u, lse_u)
        return out.view(batch_size, num_heads, head_size)


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
    """Combine two attention outputs using log-sum-exp (LSE) statistics.

    This merges two segment-wise attention results (e.g., shared and unique
    segments) into a single output that is equivalent to running attention on
    the concatenation of the segments, but without re-materializing the full
    softmax. Numerically stable via the log-sum-exp trick.

    Args:
        out_a: Attention output for segment A, shape [B, Q, H, D].
        lse_a: Log-sum-exp of (scaled) attention logits for segment A,
            shape [B, Q, H].
        out_b: Attention output for segment B, shape [B, Q, H, D].
        lse_b: Log-sum-exp of (scaled) attention logits for segment B,
            shape [B, Q, H].

    Returns:
        Tensor of shape [B, Q, H, D] with the merged attention output.

    Notes:
        - Dtypes: Inputs typically bf16; internal math is promoted to fp32.
        - Stability: Uses m = max(lse_a, lse_b) and rescales to avoid overflow.
        - Backend: If Triton is available, dispatch to a fused kernel;
          otherwise falls back to a pure PyTorch implementation.
    """
    if _TRITON_AVAILABLE:
        try:
            return _combine_lse_two_way_triton(out_a, lse_a, out_b, lse_b)
        except Exception:
            pass
    lse_a_32 = lse_a.float()
    lse_b_32 = lse_b.float()
    m = torch.maximum(lse_a_32, lse_b_32)
    adj_a = torch.exp(lse_a_32 - m)
    adj_b = torch.exp(lse_b_32 - m)
    denom = adj_a + adj_b
    denom = denom.clamp_min(torch.finfo(denom.dtype).tiny)
    out_a_32 = out_a.float()
    out_b_32 = out_b.float()
    combined = (
        out_a_32 * adj_a.unsqueeze(-1) + out_b_32 * adj_b.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    return combined.to(out_a.dtype)


def _combine_lse_many(
    outs: List[torch.Tensor],
    lses: List[torch.Tensor],
) -> torch.Tensor:
    """Generalized LSE combine for N segments.

    Args:
        outs: List of [B, Q, H, D] attention outputs per segment.
        lses: List of [B, Q, H] LSE tensors per segment.

    Returns:
        Aggregated output [B, Q, H, D].

    Notes:
        - For N==1 returns outs[0].
        - For N>=2 iteratively reduces using the two-way combine helper,
          which may dispatch to a Triton kernel when available.
    """
    assert len(outs) == len(lses) and len(outs) >= 1
    if len(outs) == 1:
        return outs[0]
    aggregated = outs[0]
    aggregated_lse = lses[0]
    for i in range(1, len(outs)):
        aggregated = _combine_lse_two_way(aggregated, aggregated_lse, outs[i], lses[i])
        # Update the running LSE as max(lse_prev, lse_i) for numerical stability.
        # This is consistent with the weighting inside two-way combine where
        # m = max(lse_prev, lse_i) per position/head.
        aggregated_lse = torch.maximum(aggregated_lse.float(), lses[i].float())
    return aggregated


if _TRITON_AVAILABLE:

    @triton.jit
    def _lse_combine_two_way_kernel(
        out1_ptr,
        out2_ptr,
        lse1_ptr,
        lse2_ptr,
        aggregated_ptr,
        bsh_stride,
        bsh,
        hdim,
        BLOCK_SIZE_BSH: tl.constexpr,
        BLOCK_SIZE_HDIM: tl.constexpr,
    ):
        """Fused two-way LSE combine over a flattened (B * Q * H, D) tile.

        Pointers:
            out1_ptr, out2_ptr: row-major matrices of shape (BSH, D).
            lse1_ptr, lse2_ptr: vectors of length BSH with per-row LSE values.
            aggregated_ptr: output matrix (BSH, D).

        Other args:
            bsh_stride: row stride (in elements) for the (BSH, D) matrices.
            bsh: total rows (B * Q * H).
            hdim: head dimension D.

        The kernel computes, per row r and feature d:
            m = max(lse1[r], lse2[r])
            adj1 = exp(lse1[r] - m); adj2 = exp(lse2[r] - m)
            aggregated[r, d] = (out1[r, d] * adj1 + out2[r, d] * adj2)
                                / (adj1 + adj2)
        """
        bsh_idx = tl.program_id(0)
        bsh_range = tl.arange(0, BLOCK_SIZE_BSH)
        hdim_range = tl.arange(0, BLOCK_SIZE_HDIM)

        lse_start = bsh_idx * BLOCK_SIZE_BSH
        lse_offs = lse_start + bsh_range

        lse1 = tl.load(lse1_ptr + lse_offs, mask=lse_offs < bsh, other=0.0)
        lse2 = tl.load(lse2_ptr + lse_offs, mask=lse_offs < bsh, other=0.0)

        max_lse = tl.maximum(lse1, lse2)
        adj1 = tl.exp(lse1 - max_lse)
        adj2 = tl.exp(lse2 - max_lse)
        denom = adj1 + adj2

        out_start = bsh_idx * BLOCK_SIZE_BSH * bsh_stride
        out_offs = out_start + (bsh_range[:, None] * bsh_stride + hdim_range[None, :])

        out1_ptrs = out1_ptr + out_offs
        out2_ptrs = out2_ptr + out_offs
        agg_ptrs = aggregated_ptr + out_offs

        for i in range(0, tl.cdiv(hdim, BLOCK_SIZE_HDIM)):
            mask = out_offs + i * BLOCK_SIZE_HDIM < hdim * bsh
            o1 = tl.load(out1_ptrs, mask=mask, other=0.0)
            o2 = tl.load(out2_ptrs, mask=mask, other=0.0)
            agg = (o1 * adj1[:, None] + o2 * adj2[:, None]) / denom[:, None]
            tl.store(agg_ptrs, agg, mask=mask)
            out1_ptrs += BLOCK_SIZE_HDIM
            out2_ptrs += BLOCK_SIZE_HDIM
            agg_ptrs += BLOCK_SIZE_HDIM


def _combine_lse_two_way_triton(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Triton-based two-way LSE combine.

    Maps [B, Q, H, D] tensors to a flattened (B*Q*H, D) layout and launches
    a fused kernel that performs numerically stable merging using the log-sum-
    exp trick. Accumulations are in fp32; the output dtype matches ``out_a``.

    Args:
        out_a: [B, Q, H, D]
        lse_a: [B, Q, H]
        out_b: [B, Q, H, D]
        lse_b: [B, Q, H]

    Returns:
        Aggregated output of shape [B, Q, H, D].
    """
    B, Q, H, D = out_a.shape
    out1_flat = out_a.contiguous().view(B * Q * H, D)
    out2_flat = out_b.contiguous().view(B * Q * H, D)
    lse1 = lse_a.contiguous().float().view(B * Q * H)
    lse2 = lse_b.contiguous().float().view(B * Q * H)
    aggregated_flat = torch.empty_like(out1_flat)

    BSH = B * Q * H
    grid = lambda META: (triton.cdiv(BSH, META["BLOCK_SIZE_BSH"]),)

    _lse_combine_two_way_kernel[grid](
        out1_flat,
        out2_flat,
        lse1,
        lse2,
        aggregated_flat,
        out1_flat.stride(0),
        BSH,
        D,
        BLOCK_SIZE_BSH=32,
        BLOCK_SIZE_HDIM=64,
        num_warps=2,
    )
    return aggregated_flat.view(B, Q, H, D)


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
