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
    """Ragged dot with group_offset support and fast path optimization."""
    if group_offset is None:
        return lax.ragged_dot(lhs, rhs, group_sizes, precision=precision, preferred_element_type=preferred_element_type)

    offset = group_offset[0]
    m, k = lhs.shape
    g_local, g_total = rhs.shape[0], group_sizes.shape[0]
    local_capacity = min(int(2.0 * m * g_local / g_total), m)

    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start, shard_end = cumsum[offset], cumsum[offset + g_local]
    num_valid = shard_end - shard_start
    local_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)

    # Global decision: use fast path only if ALL shards can use it
    can_use_fast = (num_valid <= local_capacity).astype(jnp.int32)
    all_use_fast = lax.pmin(can_use_fast, axis_name="ep") > 0

    def fast_path(_):
        jax.debug.print("FAST path: num_valid={nv}, local_capacity={lc}, m={m}", nv=num_valid, lc=local_capacity, m=m, ordered=True)
        lhs_padded = jnp.pad(lhs, ((0, local_capacity), (0, 0)))
        lhs_slice = lax.dynamic_slice(lhs_padded, (shard_start, 0), (local_capacity, k))
        adjusted = local_sizes.at[-1].add(local_capacity - num_valid)
        result = lax.ragged_dot(lhs_slice, rhs, adjusted, precision=precision, preferred_element_type=preferred_element_type)
        result = jnp.where((jnp.arange(local_capacity) < num_valid)[:, None], result, 0)
        return lax.dynamic_update_slice(jnp.zeros((m, rhs.shape[-1]), result.dtype), result, (shard_start, 0))

    def full_path(_):
        jax.debug.print("FULL path: num_valid={nv}, local_capacity={lc}, m={m}", nv=num_valid, lc=local_capacity, m=m, ordered=True)
        adjusted = local_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)
        result = lax.ragged_dot(lhs, rhs, adjusted, precision=precision, preferred_element_type=preferred_element_type)
        mask = (jnp.arange(m) >= shard_start) & (jnp.arange(m) < shard_end)
        return jnp.where(mask[:, None], result, 0)

    return lax.cond(all_use_fast, fast_path, full_path, None)


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
