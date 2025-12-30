from dataclasses import dataclass
from functools import partial

from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
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


@dataclass
class ExpertParallelInfo:
    """Information needed to reverse expert parallel dispatch."""

    original_order: jax.Array  # Indices to restore original token order after combine
    send_counts: jax.Array  # Number of tokens sent to each device
    recv_counts: jax.Array  # Number of tokens received from each device


def expert_parallel_dispatch(
    tokens: jax.Array,
    expert_indices: jax.Array,
    routing_weights: jax.Array,
    num_experts: int,
    ep_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, ExpertParallelInfo]:
    """Dispatch tokens to devices via all-to-all based on expert assignment.

    Args:
        tokens: Token hidden states (num_tokens, hidden_size)
        expert_indices: Global expert ID for each token (num_tokens,)
        routing_weights: Softmax weights for each token (num_tokens,)
        num_experts: Total number of experts across all devices
        ep_size: Number of expert parallel devices

    Returns:
        local_tokens: Tokens after all-to-all redistribution
        local_expert_ids: Local expert indices (0 to experts_per_device-1)
        local_weights: Routing weights after redistribution
        info: ExpertParallelInfo for combining results
    """
    experts_per_device = num_experts // ep_size

    # Compute target device and local expert ID
    target_device = expert_indices // experts_per_device
    local_expert_id = expert_indices % experts_per_device

    # Sort tokens by target device for all-to-all
    sort_idx = jnp.argsort(target_device)
    tokens_sorted = tokens[sort_idx]
    local_expert_id_sorted = local_expert_id[sort_idx]
    weights_sorted = routing_weights[sort_idx]

    # Count tokens going to each device
    send_counts = jnp.zeros(ep_size, dtype=jnp.int32)
    for i in range(ep_size):
        send_counts = send_counts.at[i].set(jnp.sum(target_device == i))

    # Compute receive counts (symmetric in this implementation)
    # In practice, each device receives what others send to it
    recv_counts = send_counts  # Will be updated by all-to-all

    # All-to-all: redistribute tokens to their target devices
    # Split along batch dim, concat along batch dim
    received_tokens = jax.lax.all_to_all(
        tokens_sorted, axis_name="ep", split_axis=0, concat_axis=0, tiled=True
    )
    received_expert_ids = jax.lax.all_to_all(
        local_expert_id_sorted, axis_name="ep", split_axis=0, concat_axis=0, tiled=True
    )
    received_weights = jax.lax.all_to_all(
        weights_sorted, axis_name="ep", split_axis=0, concat_axis=0, tiled=True
    )

    # Exchange send_counts to get recv_counts
    recv_counts = jax.lax.all_to_all(
        send_counts, axis_name="ep", split_axis=0, concat_axis=0, tiled=True
    )

    info = ExpertParallelInfo(
        original_order=jnp.argsort(sort_idx),  # To restore original order
        send_counts=send_counts,
        recv_counts=recv_counts,
    )

    return received_tokens, received_expert_ids, received_weights, info


def expert_parallel_combine(
    local_outputs: jax.Array,
    info: ExpertParallelInfo,
) -> jax.Array:
    """Combine expert outputs via reverse all-to-all.

    Args:
        local_outputs: Outputs from local expert computation
        info: ExpertParallelInfo from dispatch

    Returns:
        outputs: Combined outputs in original token order
    """
    # Reverse all-to-all: send outputs back to original devices
    gathered_outputs = jax.lax.all_to_all(
        local_outputs, axis_name="ep", split_axis=0, concat_axis=0, tiled=True
    )

    # Restore original token order
    return gathered_outputs[info.original_order]


def create_expert_parallel_fn(
    expert_forward_fn,
    num_experts: int,
    experts_per_device: int,
    mesh: jax.sharding.Mesh,
):
    """Create a shardmap-wrapped function for expert parallel computation.

    Args:
        expert_forward_fn: Function that takes (tokens, group_sizes) and returns outputs
        num_experts: Total number of experts
        experts_per_device: Number of experts per device
        mesh: The device mesh with "ep" axis

    Returns:
        A function that handles expert parallel dispatch, compute, and combine
    """
    ep_size = num_experts // experts_per_device

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(None, None), P(None), P(None)),  # tokens, expert_indices, weights
        out_specs=P(None, None),  # output tokens
        check_rep=False,
    )
    def ep_forward(tokens, expert_indices, routing_weights):
        # Dispatch tokens to correct devices
        local_tokens, local_expert_ids, local_weights, info = expert_parallel_dispatch(
            tokens, expert_indices, routing_weights, num_experts, ep_size
        )

        # Sort by local expert for ragged_dot
        local_sorted, local_group_sizes, local_unsort, _ = prepare_routing(
            local_tokens, local_expert_ids, experts_per_device
        )

        # Apply expert computation
        local_outputs_sorted = expert_forward_fn(local_sorted, local_group_sizes)

        # Unsort local outputs
        local_outputs = local_outputs_sorted[local_unsort]

        # Apply routing weights
        local_outputs = local_outputs * local_weights[:, None]

        # Combine outputs via reverse all-to-all
        return expert_parallel_combine(local_outputs, info)

    return ep_forward
