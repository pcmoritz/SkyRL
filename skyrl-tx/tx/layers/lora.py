import torch
import torch.nn as nn
from typing import Callable

from tx.layers.util import Param, prepare_routing


def grouped_mm(x: torch.Tensor, weights: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    """Grouped matrix multiplication with CPU fallback.

    Args:
        x: Input tensor [N, in_features] where tokens are sorted by group
        weights: Weight tensor [num_groups, in_features, out_features]
        group_sizes: Number of tokens per group [num_groups]

    Returns:
        Output tensor [N, out_features]
    """
    # Use native grouped_mm on GPU/MPS
    if x.device.type in ['cuda', 'mps']:
        return torch._grouped_mm(x, weights, group_sizes)

    # CPU fallback: loop over groups
    outputs = []
    start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        group_size = int(group_size)
        if group_size > 0:
            x_group = x[start_idx:start_idx + group_size]  # [group_size, in_features]
            w_group = weights[group_idx]  # [in_features, out_features]
            output_group = x_group @ w_group  # [group_size, out_features]
            outputs.append(output_group)
            start_idx += group_size

    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, weights.shape[-1], device=x.device, dtype=x.dtype)


class LoRAMixin:
    """A mixin for PyTorch modules to add multi-adapter LoRA support.
    This mixin adds LoRA parameters (lora_A, lora_B) and methods to apply
    the low-rank adaptation to a base module's output. It is designed to
    be used with layers like nn.Linear.
    """

    lora_scaling: torch.Tensor | None
    lora_ranks: torch.Tensor | None
    lora_A: nn.Parameter | None
    lora_B: nn.Parameter | None
    max_lora_adapters: int
    max_lora_rank: int

    def init_lora(
        self,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank

        if max_lora_adapters == 0:
            self.lora_scaling = None
            self.lora_ranks = None
            self.lora_A = None
            self.lora_B = None
        else:
            # Register as buffers (non-trainable by default)
            self.register_buffer('lora_scaling', torch.full((max_lora_adapters,), 1.0, dtype=dtype, device=device))
            self.register_buffer('lora_ranks', torch.full((max_lora_adapters,), max_lora_rank, dtype=torch.int32, device=device))

            # Initialize LoRA matrices
            lora_A = torch.empty(*shape_A, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(lora_A)
            self.lora_A = nn.Parameter(lora_A)

            lora_B = torch.zeros(*shape_B, dtype=dtype, device=device)
            self.lora_B = nn.Parameter(lora_B)

    def apply_lora(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        adapter_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.max_lora_adapters == 0 or adapter_indices is None:
            return base_output

        (batch_size, seq_len, in_features) = x.shape
        assert len(self.lora_A.shape) == 3 and self.lora_A.shape[1] == in_features
        assert adapter_indices.shape[0] == batch_size

        x_flat = x.reshape(-1, in_features)
        adapter_indices_expanded = adapter_indices.repeat_interleave(seq_len)

        # Sort tokens to prepare for grouped_mm
        x_sorted, group_sizes, unsort_indices, _ = prepare_routing(
            x_flat, adapter_indices_expanded, self.max_lora_adapters
        )

        # Apply LoRA using grouped_mm: x @ A @ B
        intermediate = grouped_mm(x_sorted, self.lora_A, group_sizes)
        lora_output_sorted = grouped_mm(intermediate, self.lora_B, group_sizes)

        # Unsort, reshape, scale
        lora_output = lora_output_sorted[unsort_indices].reshape(batch_size, seq_len, -1)
        lora_output = lora_output * self.lora_scaling[adapter_indices, None, None]
        return base_output + lora_output.reshape(base_output.shape)


class LoRALinear(LoRAMixin, nn.Linear):
    """An nn.Linear layer with multi-adapter LoRA support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: torch.dtype = torch.float32,
        param_dtype: torch.dtype | None = None,
        use_bias: bool = True,
        device: torch.device | None = None,
    ) -> None:
        param_dtype = param_dtype or dtype

        super().__init__(
            in_features,
            out_features,
            bias=use_bias,
            dtype=param_dtype,
            device=device,
        )

        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, max_lora_rank, out_features),
            dtype=param_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        base_out = super().forward(x)
        return self.apply_lora(x, base_out, adapter_indices)


class LoRAExpert(LoRAMixin, nn.Module):
    """Expert layer with multi-adapter LoRA support."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        max_lora_adapters: int = 0,
        max_lora_rank: int = 8,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # Initialize expert weights
        weight = torch.empty(num_experts, in_features, out_features, dtype=dtype, device=device)
        nn.init.kaiming_normal_(weight)
        self.weight = nn.Parameter(weight)

        self.init_lora(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, num_experts, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, num_experts, max_lora_rank, out_features),
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        group_sizes: torch.Tensor,
        adapter_indices_sorted: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_out = grouped_mm(x, self.weight, group_sizes)

        if self.max_lora_adapters == 0 or adapter_indices_sorted is None:
            return base_out

        # Reconstruct expert indices from group_sizes
        expert_indices = torch.repeat_interleave(
            torch.arange(self.num_experts, device=x.device),
            group_sizes
        )

        # Flatten (adapter, expert) into a single routing dimension.
        flattened_indices = adapter_indices_sorted * self.num_experts + expert_indices
        num_flattened_groups = self.max_lora_adapters * self.num_experts

        # Reshape lora_A and lora_B to merge (max_lora_adapters, num_experts) dimensions
        lora_A_reshaped = self.lora_A.reshape(num_flattened_groups, self.in_features, self.max_lora_rank)
        lora_B_reshaped = self.lora_B.reshape(num_flattened_groups, self.max_lora_rank, self.out_features)

        # Sort tokens by combined index
        x_sorted, combined_group_sizes, unsort_indices, _ = prepare_routing(x, flattened_indices, num_flattened_groups)

        # Apply LoRA using grouped_mm: x @ A @ B
        intermediate = grouped_mm(x_sorted, lora_A_reshaped, combined_group_sizes)
        lora_output_sorted = grouped_mm(intermediate, lora_B_reshaped, combined_group_sizes)

        # Unsort and apply scaling
        lora_output = lora_output_sorted[unsort_indices]
        lora_output = lora_output * self.lora_scaling[adapter_indices_sorted, None]

        return base_out + lora_output


def update_adapter_config(model: nn.Module, adapter_index: int, lora_rank: int, lora_alpha: float):
    """Update lora_ranks and lora_scaling for a specific adapter across all LoRA layers.

    Note: This method needs to be called BEFORE any training happens, you should not update
    the config for the same adapter index multiple times throughout training (e.g. it will
    invalidate your current training progress and also violate the assumption that lora_B
    is zero).

    Args:
        model: The model containing LoRA layers
        adapter_index: Index of the adapter to update
        lora_rank: Rank to set for this adapter
        lora_alpha: Alpha value to use for computing scaling (alpha / rank)
    """
    scaling = lora_alpha / lora_rank

    for name, module in model.named_modules():
        if isinstance(module, LoRAMixin) and hasattr(module, 'lora_ranks'):
            if module.lora_ranks is not None:
                module.lora_ranks[adapter_index] = lora_rank
            if module.lora_scaling is not None:
                module.lora_scaling[adapter_index] = scaling
            if module.lora_A is not None:
                # Zero out columns beyond the rank for this adapter; lora_B is already zero
                with torch.no_grad():
                    module.lora_A.data[adapter_index, :, lora_rank:] = 0.0
