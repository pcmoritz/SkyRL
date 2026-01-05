from functools import partial
from typing import Callable

from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


def prepare_routing(
    tokens: jax.Array, indices: jax.Array, num_groups: int, adapter_indices: jax.Array | None = None
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Prepare inputs for ragged_dot operations by sorting tokens by group.

    Args:
        tokens: Array of shape (num_tokens, ...) to be sorted by group
        indices: Array of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional array of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after ragged operations
    """
    sort_indices = jnp.argsort(indices)
    sorted_tokens = tokens[sort_indices]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_indices]
    group_sizes = jnp.bincount(indices, length=num_groups)
    unsort_indices = jnp.argsort(sort_indices)
    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices


def _local_expert_computation(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array],
    num_experts: int,
    num_experts_per_tok: int,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Compute expert outputs locally (EP=1 path).

    This handles the case where all experts are on a single device, no cross-device
    communication is needed.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_size]
        selected_experts: Expert assignments [num_tokens, num_experts_per_tok]
        routing_weights: Routing weights [num_tokens, num_experts_per_tok]
        expert_fn: Function that takes (sorted_tokens, group_sizes, adapter_indices)
                   and returns expert outputs
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts per token (top-k)
        adapter_indices: Optional adapter indices [num_tokens]

    Returns:
        Combined expert outputs [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.shape[0]

    # Expand tokens for each selected expert
    selected_experts_flat = selected_experts.ravel()
    hidden_states_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
    adapter_indices_expanded = (
        jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
    )

    # Sort by expert assignment for efficient batched computation
    hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
        hidden_states_expanded,
        selected_experts_flat,
        num_experts,
        adapter_indices=adapter_indices_expanded,
    )

    # Apply expert function
    expert_output_sorted = expert_fn(hidden_states_sorted, group_sizes, adapter_indices_sorted)

    # Unsort and combine with routing weights
    expert_output = expert_output_sorted[unsort_indices]
    expert_output = expert_output.reshape(num_tokens, num_experts_per_tok, -1)
    return jnp.sum(expert_output * routing_weights[..., None], axis=1)


def expert_parallel_dispatch_combine(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array],
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Dispatch tokens to experts across EP devices and combine results.

    Uses shard_map to distribute computation across expert parallel devices.
    Each EP device handles num_experts/ep_size experts.

    Args:
        hidden_states: Input tokens [num_tokens, hidden_size]
        selected_experts: Expert assignments [num_tokens, num_experts_per_tok]
        routing_weights: Routing weights [num_tokens, num_experts_per_tok]
        expert_fn: Function that takes (sorted_tokens, group_sizes, adapter_indices)
                   and returns expert outputs. Expert weights should be sharded on
                   the first dimension across the "ep" axis.
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts per token (top-k)
        hidden_size: Hidden dimension size
        adapter_indices: Optional adapter indices [num_tokens]

    Returns:
        Combined expert outputs [num_tokens, hidden_size]
    """
    mesh = jax.get_mesh()
    ep_size = mesh.shape.get("ep", 1)

    # Fall back to local computation if EP=1
    if ep_size == 1:
        return _local_expert_computation(
            hidden_states,
            selected_experts,
            routing_weights,
            expert_fn,
            num_experts,
            num_experts_per_tok,
            adapter_indices,
        )

    experts_per_rank = num_experts // ep_size

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P(), P() if adapter_indices is not None else P()),
        out_specs=P(),
        check_rep=False,
    )
    def _ep_expert_fn(hidden_states, selected_experts, routing_weights, adapter_indices):
        num_tokens = hidden_states.shape[0]
        ep_idx = jax.lax.axis_index("ep")

        # Expand for top-k experts
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
        routing_weights_flat = routing_weights.ravel()
        adapter_indices_expanded = (
            jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
        )

        # Determine which tokens belong to this EP rank
        local_expert_start = ep_idx * experts_per_rank
        local_expert_end = local_expert_start + experts_per_rank
        is_local = (selected_experts_flat >= local_expert_start) & (selected_experts_flat < local_expert_end)

        # Map global expert indices to local indices
        local_expert_indices = jnp.where(is_local, selected_experts_flat - local_expert_start, 0)

        # Sort tokens by local expert assignment
        # Non-local tokens get assigned to a dummy high index for sorting
        sort_key = jnp.where(is_local, local_expert_indices, experts_per_rank + 1)
        sort_indices = jnp.argsort(sort_key)
        unsort_indices = jnp.argsort(sort_indices)

        # Sort everything
        hidden_sorted = hidden_states_expanded[sort_indices]
        local_experts_sorted = local_expert_indices[sort_indices]
        weights_sorted = routing_weights_flat[sort_indices]
        adapter_sorted = adapter_indices_expanded[sort_indices] if adapter_indices_expanded is not None else None
        is_local_sorted = is_local[sort_indices]

        # Compute group sizes for local experts only using masked bincount
        # Set non-local tokens to an out-of-range index so they don't contribute
        masked_indices = jnp.where(is_local_sorted, local_experts_sorted, experts_per_rank)
        group_sizes = jnp.bincount(masked_indices, length=experts_per_rank + 1)[:experts_per_rank]

        # Apply expert computation (only on local tokens, but we process all for simplicity)
        expert_output_sorted = expert_fn(hidden_sorted, group_sizes, adapter_sorted)

        # Zero out non-local outputs
        expert_output_sorted = jnp.where(is_local_sorted[:, None], expert_output_sorted, 0.0)

        # Apply routing weights
        expert_output_sorted = expert_output_sorted * weights_sorted[:, None]

        # Unsort
        expert_output = expert_output_sorted[unsort_indices]

        # Reshape and sum over top-k dimension
        expert_output = expert_output.reshape(num_tokens, num_experts_per_tok, hidden_size)
        local_output = jnp.sum(expert_output, axis=1)

        # All-reduce across EP devices to combine outputs from all experts
        return jax.lax.psum(local_output, "ep")

    return _ep_expert_fn(
        hidden_states,
        selected_experts,
        routing_weights,
        adapter_indices if adapter_indices is not None else jnp.zeros(hidden_states.shape[0], dtype=jnp.int32),
    )
