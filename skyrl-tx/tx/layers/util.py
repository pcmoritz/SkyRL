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
    """Ragged dot product with group_offset support.

    When group_offset is specified, rhs contains groups [offset, offset + g_local).
    Tokens outside this range are routed to boundary groups and masked to zero.

    Optimization: When the number of tokens for local experts is within 1.2x of
    the expected amount (CAPACITY * g_local / g_total), uses a fast path that
    only computes local_capacity tokens instead of all m tokens.
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
    g_total = group_sizes.shape[0]
    n = rhs.shape[-1]

    assert g_local > 0, "rhs must have at least one group"

    # Static local capacity with 1.2x factor, capped at m
    local_capacity = min(int(1.2 * m * g_local / g_total), m)

    # Compute token boundaries for local groups
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]
    num_valid = shard_end - shard_start

    local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)

    def fast_path(_):
        """Fast path: compute only local_capacity tokens when token count is small."""
        # Dynamic slice starting at shard_start
        lhs_slice = lax.dynamic_slice(lhs, (shard_start, 0), (local_capacity, lhs.shape[1]))
        # Adjust last group to absorb extra capacity beyond num_valid
        adjusted_sizes = local_group_sizes.at[-1].add(local_capacity - num_valid)
        result_slice = lax.ragged_dot(
            lhs_slice,
            rhs,
            adjusted_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        # Mask out tokens beyond num_valid
        token_idx = jnp.arange(local_capacity)
        valid_mask = token_idx < num_valid
        result_slice = jnp.where(valid_mask[:, None], result_slice, 0)
        # Create output and place result at shard_start
        output = jnp.zeros((m, n), dtype=result_slice.dtype)
        return lax.dynamic_update_slice(output, result_slice, (shard_start, 0))

    def full_path(_):
        """Full path: compute all m tokens with boundary absorption."""
        token_idx = jnp.arange(m)
        valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)
        adjusted_sizes = local_group_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)
        result = lax.ragged_dot(
            lhs,
            rhs,
            adjusted_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        return jnp.where(valid_mask[:, None], result, 0)

    return lax.cond(num_valid <= local_capacity, fast_path, full_path, None)


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
