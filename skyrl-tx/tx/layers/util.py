from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec


DEFAULT_RAGGED_DOT_BUCKET_SIZES = (128, 256, 512, 1024, 2048, 4096)


def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    precision=None,
    preferred_element_type=None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
    """Ragged dot product with group_offset support.

    When group_offset is specified, rhs contains groups [offset, offset + g_local).
    Tokens outside this range are routed to boundary groups and masked to zero.
    """
    if group_offset is None:
        return lax.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    assert group_offset.shape == (1,), "group_offset must have shape (1,)"
    offset = group_offset[0]
    m = lhs.shape[0]
    g_local = rhs.shape[0]
    n = rhs.shape[2]

    assert g_local > 0, "rhs must have at least one group"

    # Compute token boundaries for local groups
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]
    local_len = shard_end - shard_start

    bucket_sizes = tuple(size for size in DEFAULT_RAGGED_DOT_BUCKET_SIZES if size < m) + (m,)
    bucket_sizes_array = jnp.array(bucket_sizes, dtype=local_len.dtype)
    bucket_index = jnp.sum(local_len > bucket_sizes_array)

    def _bucketed_result(bucket_len: int) -> jax.Array:
        max_start = m - bucket_len
        window_start = jnp.minimum(shard_start, max_start)
        window_start = jnp.maximum(window_start, 0)
        window_end = window_start + bucket_len
        prefix = shard_start - window_start
        suffix = window_end - shard_end

        local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
        adjusted_group_sizes = local_group_sizes.at[0].add(prefix).at[-1].add(suffix)

        window_lhs = lax.dynamic_slice_in_dim(lhs, window_start, bucket_len, axis=0)
        window_result = lax.ragged_dot(
            window_lhs,
            rhs,
            adjusted_group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

        window_idx = jnp.arange(bucket_len, dtype=shard_start.dtype) + window_start
        window_valid = (window_idx >= shard_start) & (window_idx < shard_end)
        window_result = jnp.where(window_valid[:, None], window_result, 0)

        out = jnp.zeros((m, n), dtype=window_result.dtype)
        return lax.dynamic_update_slice(out, window_result, (window_start, 0))

    def _branch(bucket_len: int):
        def _run(_):
            return _bucketed_result(bucket_len)

        return _run

    branches = [_branch(size) for size in bucket_sizes]
    dummy = jnp.array(0, dtype=local_len.dtype)
    return lax.switch(bucket_index, branches, dummy)


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
