from typing import Optional, List

import torch


class InputMetadata:
    """Metadata for input sequences. Used in PagedAttention.

    Args:
        prompt_lens: Lengths of prompts.
        slot_mapping: The address to write the new KV to of each token.
        max_context_len: The maximum context length.
        context_lens: the length of attention context for each sequence.
        block_tables: The block tables. (Seq id -> list of physical block)
        shared_ks: Optional list of shared K tensors for prefill (each
            tensor shaped [shared_batch, shared_len, num_kv_heads, head_dim]).
        shared_vs: Optional list of shared V tensors matching shared_ks.
        shared_max_lens: Optional list of max sequence lengths per shared
            segment (padded shared lengths), used to construct masks.
    """

    def __init__(
        self,
        is_prompt: bool,
        slot_mapping: torch.Tensor,
        max_context_len: Optional[int],
        context_lens: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        use_cuda_graph: bool,
        shared_ks: Optional[List[torch.Tensor]] = None,
        shared_vs: Optional[List[torch.Tensor]] = None,
        shared_max_lens: Optional[List[int]] = None,
        shared_prefix_len: Optional[int] = None,
        shared_groups: Optional[int] = None,
    ) -> None:
        self.is_prompt = is_prompt
        self.max_context_len = max_context_len
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.use_cuda_graph = use_cuda_graph

        # Optional Hydragen-style shared segments for prefill path.
        # Expected shapes per segment (padded):
        #   Ks[i], Vs[i]: [shared_batch, shared_len, num_kv_heads, head_dim]
        # When provided, prefill attention may compute per-segment attention
        # and merge via LSE-combine.
        self.shared_ks = shared_ks
        self.shared_vs = shared_vs
        self.shared_max_lens = shared_max_lens
        self.shared_prefix_len = shared_prefix_len
        self.shared_groups = shared_groups

        # Set during the execution of the first attention op.
        # FIXME(woosuk): This is a hack.
        self.attn_bias = None

    def __repr__(self) -> str:
        return (
            "InputMetadata("
            f"is_prompt={self.is_prompt}, "
            f"max_context_len={self.max_context_len}, "
            f"slot_mapping={self.slot_mapping}, "
            f"context_lens={self.context_lens}, "
            f"block_tables={self.block_tables}, "
            f"use_cuda_graph={self.use_cuda_graph}, "
            f"shared_ks={'set' if self.shared_ks is not None else 'None'}, "
            f"shared_vs={'set' if self.shared_vs is not None else 'None'}, "
            f"shared_max_lens={self.shared_max_lens}, "
            f"shared_prefix_len={self.shared_prefix_len}, "
            f"shared_groups={self.shared_groups})"
        )
