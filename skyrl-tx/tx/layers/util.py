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
    """Dispatch tokens to experts across EP ranks, compute, and combine results.

    This function implements expert parallelism by:
    1. Sorting tokens by destination EP rank
    2. Using all-to-all to exchange tokens between ranks
    3. Computing local experts on received tokens
    4. Using reverse all-to-all to return results
    5. Applying routing weights and summing

    Args:
        hidden_states: Input tokens [num_tokens, hidden_size]
        selected_experts: Expert assignments [num_tokens, num_experts_per_tok]
        routing_weights: Routing weights [num_tokens, num_experts_per_tok]
        expert_fn: Function to apply experts, takes (tokens, group_sizes, adapter_indices)
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts each token routes to
        hidden_size: Hidden dimension size
        adapter_indices: Optional adapter indices [num_tokens]

    Returns:
        Combined expert outputs [num_tokens, hidden_size]
    """
    mesh = jax.get_mesh()
    ep_size = mesh.shape.get("ep", 1)

    if ep_size == 1:
        # No EP - use local computation path
        return _local_expert_computation(
            hidden_states, selected_experts, routing_weights, expert_fn, num_experts, num_experts_per_tok, adapter_indices
        )

    experts_per_rank = num_experts // ep_size

    # Expand hidden states and adapter indices for each expert selection
    num_tokens = hidden_states.shape[0]
    hidden_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
    selected_flat = selected_experts.ravel()

    if adapter_indices is not None:
        adapter_expanded = jnp.repeat(adapter_indices, num_experts_per_tok)
    else:
        adapter_expanded = None

    # Compute destination EP rank and local expert index for each token-expert pair
    dest_ranks = selected_flat // experts_per_rank
    local_expert_ids = selected_flat % experts_per_rank

    # Sort by destination rank
    sort_indices = jnp.argsort(dest_ranks)
    sorted_hidden = hidden_expanded[sort_indices]
    sorted_local_experts = local_expert_ids[sort_indices]
    sorted_adapter = None if adapter_expanded is None else adapter_expanded[sort_indices]
    unsort_indices = jnp.argsort(sort_indices)

    # Count tokens going to each rank
    send_counts = jnp.bincount(dest_ranks, length=ep_size)

    # Pad to uniform size for all-to-all (each rank sends/receives same amount)
    max_per_rank = num_tokens * num_experts_per_tok  # Upper bound
    tokens_per_rank = max_per_rank // ep_size + (1 if max_per_rank % ep_size else 0)
    total_padded = tokens_per_rank * ep_size

    # Pad inputs
    sorted_hidden_padded = jnp.pad(
        sorted_hidden, ((0, total_padded - sorted_hidden.shape[0]), (0, 0)), mode="constant"
    )
    sorted_local_experts_padded = jnp.pad(
        sorted_local_experts, (0, total_padded - sorted_local_experts.shape[0]), constant_values=-1
    )
    if sorted_adapter is not None:
        sorted_adapter_padded = jnp.pad(
            sorted_adapter, (0, total_padded - sorted_adapter.shape[0]), constant_values=0
        )
    else:
        sorted_adapter_padded = None

    # Use shard_map for explicit all-to-all communication
    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("ep", None), P("ep"), P("ep") if sorted_adapter_padded is not None else P(), P("ep")),
        out_specs=P("ep", None),
        check_rep=False,
    )
    def _sharded_expert_computation(hidden_chunk, expert_ids_chunk, adapter_chunk, send_counts_chunk):
        # All-to-all: exchange tokens with other EP ranks
        # Each rank sends tokens_per_rank tokens to each other rank
        received_hidden = jax.lax.all_to_all(
            hidden_chunk.reshape(ep_size, tokens_per_rank, hidden_size),
            axis_name="ep",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(-1, hidden_size)

        received_expert_ids = jax.lax.all_to_all(
            expert_ids_chunk.reshape(ep_size, tokens_per_rank),
            axis_name="ep",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).ravel()

        if adapter_chunk is not None:
            received_adapter = jax.lax.all_to_all(
                adapter_chunk.reshape(ep_size, tokens_per_rank),
                axis_name="ep",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ).ravel()
        else:
            received_adapter = None

        # Gather send counts from all ranks to know receive counts
        all_send_counts = jax.lax.all_gather(send_counts_chunk, axis_name="ep")
        ep_rank = jax.lax.axis_index("ep")
        receive_counts = all_send_counts[:, ep_rank]  # Column for this rank

        # Mask out padded tokens (expert_id == -1)
        valid_mask = received_expert_ids >= 0
        num_valid = valid_mask.sum()

        # Prepare for local expert computation
        # Sort received tokens by local expert id for ragged_dot
        received_sorted, local_group_sizes, local_unsort, received_adapter_sorted = prepare_routing(
            received_hidden, received_expert_ids, experts_per_rank, adapter_indices=received_adapter
        )

        # Apply local experts
        expert_output = expert_fn(received_sorted, local_group_sizes, received_adapter_sorted)

        # Unsort to match received order
        expert_output_unsorted = expert_output[local_unsort]

        # Reverse all-to-all: send results back
        returned_output = jax.lax.all_to_all(
            expert_output_unsorted.reshape(ep_size, tokens_per_rank, hidden_size),
            axis_name="ep",
            split_axis=0,
            concat_axis=0,
            tiled=True,
        ).reshape(-1, hidden_size)

        return returned_output

    # Handle the case where adapter_indices is None
    if sorted_adapter_padded is None:
        # Create dummy adapter array for shard_map signature consistency
        dummy_adapter = jnp.zeros((total_padded,), dtype=jnp.int32)

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P("ep", None), P("ep"), P("ep")),
            out_specs=P("ep", None),
            check_rep=False,
        )
        def _sharded_expert_computation_no_adapter(hidden_chunk, expert_ids_chunk, send_counts_chunk):
            received_hidden = jax.lax.all_to_all(
                hidden_chunk.reshape(ep_size, tokens_per_rank, hidden_size),
                axis_name="ep",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ).reshape(-1, hidden_size)

            received_expert_ids = jax.lax.all_to_all(
                expert_ids_chunk.reshape(ep_size, tokens_per_rank),
                axis_name="ep",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ).ravel()

            received_sorted, local_group_sizes, local_unsort, _ = prepare_routing(
                received_hidden, received_expert_ids, experts_per_rank, adapter_indices=None
            )

            expert_output = expert_fn(received_sorted, local_group_sizes, None)
            expert_output_unsorted = expert_output[local_unsort]

            returned_output = jax.lax.all_to_all(
                expert_output_unsorted.reshape(ep_size, tokens_per_rank, hidden_size),
                axis_name="ep",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ).reshape(-1, hidden_size)

            return returned_output

        output_padded = _sharded_expert_computation_no_adapter(
            sorted_hidden_padded, sorted_local_experts_padded, send_counts
        )
    else:
        output_padded = _sharded_expert_computation(
            sorted_hidden_padded, sorted_local_experts_padded, sorted_adapter_padded, send_counts
        )

    # Remove padding and unsort
    output_unpadded = output_padded[: num_tokens * num_experts_per_tok]
    output_unsorted = output_unpadded[unsort_indices]

    # Reshape and apply routing weights
    output_reshaped = output_unsorted.reshape(num_tokens, num_experts_per_tok, hidden_size)
    return jnp.sum(output_reshaped * routing_weights[..., None], axis=1)


def _local_expert_computation(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array],
    num_experts: int,
    num_experts_per_tok: int,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Local expert computation without EP (original path)."""
    num_tokens = hidden_states.shape[0]
    selected_flat = selected_experts.ravel()
    hidden_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)

    if adapter_indices is not None:
        adapter_expanded = jnp.repeat(adapter_indices, num_experts_per_tok)
    else:
        adapter_expanded = None

    sorted_hidden, group_sizes, unsort_indices, sorted_adapter = prepare_routing(
        hidden_expanded, selected_flat, num_experts, adapter_indices=adapter_expanded
    )

    expert_output = expert_fn(sorted_hidden, group_sizes, sorted_adapter)
    unsorted_output = expert_output[unsort_indices]
    output_reshaped = unsorted_output.reshape(num_tokens, num_experts_per_tok, hidden_states.shape[-1])
    return jnp.sum(output_reshaped * routing_weights[..., None], axis=1)
