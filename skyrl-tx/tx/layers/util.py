from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec


def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    precision=None,
    preferred_element_type=None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
    """Ragged dot with group_offset support and chunked processing optimization."""
    if group_offset is None:
        return lax.ragged_dot(lhs, rhs, group_sizes, precision=precision, preferred_element_type=preferred_element_type)

    offset = group_offset[0]
    m, k = lhs.shape
    g_local, g_total = rhs.shape[0], group_sizes.shape[0]
    local_capacity = min(int(3.0 * m * g_local / g_total), m)
    n = rhs.shape[-1]

    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start, shard_end = cumsum[offset], cumsum[offset + g_local]
    num_valid = shard_end - shard_start
    local_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
    local_cumsum = jnp.cumulative_sum(local_sizes, include_initial=True)

    def process_chunk(carry):
        chunk_idx, result = carry

        # This chunk covers local tokens [local_start, local_end)
        local_start = chunk_idx * local_capacity
        local_end = jnp.minimum(local_start + local_capacity, num_valid)
        chunk_size = local_end - local_start

        # Global position in lhs
        global_start = shard_start + local_start

        # Compute group sizes for this chunk
        chunk_group_sizes = jnp.clip(
            jnp.minimum(local_cumsum[1:], local_end) - jnp.maximum(local_cumsum[:-1], local_start),
            0, None
        )

        # Handle dynamic_slice clamping
        clamped_start = jnp.minimum(global_start, m - local_capacity)
        offset_in_slice = global_start - clamped_start

        lhs_slice = lax.dynamic_slice(lhs, (global_start, 0), (local_capacity, k))

        # Adjust group sizes: absorb prefix and suffix
        adjusted = chunk_group_sizes.at[0].add(offset_in_slice).at[-1].add(local_capacity - offset_in_slice - chunk_size)

        chunk_result = lax.ragged_dot(lhs_slice, rhs, adjusted, precision=precision, preferred_element_type=preferred_element_type)

        # Mask to keep only valid tokens
        idx = jnp.arange(local_capacity)
        chunk_result = jnp.where(((idx >= offset_in_slice) & (idx < offset_in_slice + chunk_size))[:, None], chunk_result, 0)

        # Update result at correct position
        result = lax.dynamic_update_slice(result, chunk_result, (clamped_start, 0))

        return (chunk_idx + 1, result)

    def continue_loop(carry):
        chunk_idx, _ = carry
        return chunk_idx * local_capacity < num_valid

    init = (jnp.array(0, dtype=jnp.int32), jnp.zeros((m, n), dtype=lhs.dtype))
    _, result = lax.while_loop(continue_loop, process_chunk, init)
    return result


def get_local_capacity(m: int, g_local: int, g_total: int) -> int:
    """Get the local capacity for fast path."""
    return min(int(3.0 * m * g_local / g_total), m)


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


def shard_map_ep(module: nnx.Module, func, *args):
    """Apply shard_map over the 'ep' axis for a stateful nnx.Module.

    Args:
        module: The NNX module (will be split into graph/state).
        func: Function to run inside shard_map. Signature: (module, *args).
        *args: Arguments to pass to func (replicated across shards).
    """
    graphdef, state = nnx.split(module)
    # Extract only 'ep' dims from PartitionSpecs, replacing others with None
    state_specs = jax.tree.map(
        lambda s: PartitionSpec(*(p if p == "ep" else None for p in s)) if isinstance(s, PartitionSpec) else s,
        nnx.get_partition_spec(state),
        is_leaf=lambda x: isinstance(x, PartitionSpec),
    )
    in_specs = (state_specs,) + (PartitionSpec(),) * len(args)

    @jax.shard_map(mesh=get_abstract_mesh(), in_specs=in_specs, out_specs=PartitionSpec(), axis_names={"ep"})
    def _body(state, *fn_args):
        module_shard = nnx.merge(graphdef, state)
        return func(module_shard, *fn_args)

    return _body(state, *args)
