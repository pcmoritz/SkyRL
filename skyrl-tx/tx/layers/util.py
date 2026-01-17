from flax import nnx
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec

_CUBLAS_GROUPED_GEMM_TARGET = "cublas_gemm_grouped_batched_ex"
_CUBLAS_GROUPED_GEMM_AVAILABLE: bool | None = None


def _normalize_group_offset(group_offset: jax.Array | None) -> jax.Array | None:
    if group_offset is None:
        return None
    group_offset = jnp.asarray(group_offset, dtype=jnp.int32)
    if group_offset.shape == ():
        return group_offset[None]
    if group_offset.shape != (1,):
        raise ValueError(f"group_offset must have shape () or (1,), got {group_offset.shape}.")
    return group_offset


def _cublas_grouped_gemm_available() -> bool:
    global _CUBLAS_GROUPED_GEMM_AVAILABLE
    if _CUBLAS_GROUPED_GEMM_AVAILABLE is not None:
        return _CUBLAS_GROUPED_GEMM_AVAILABLE
    try:
        from tx.ffi import cublas_grouped_gemm
    except Exception:
        _CUBLAS_GROUPED_GEMM_AVAILABLE = False
        return False
    try:
        _CUBLAS_GROUPED_GEMM_AVAILABLE = cublas_grouped_gemm.register()
    except Exception:
        _CUBLAS_GROUPED_GEMM_AVAILABLE = False
    return _CUBLAS_GROUPED_GEMM_AVAILABLE


def _ragged_dot_cublas(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    preferred_element_type,
) -> jax.Array | None:
    if jax.default_backend() != "gpu":
        return None
    if not _cublas_grouped_gemm_available():
        return None

    out_dtype = preferred_element_type or jnp.result_type(lhs, rhs)
    result_shape = jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[2]), out_dtype)
    call = jax.ffi.ffi_call(
        _CUBLAS_GROUPED_GEMM_TARGET,
        result_shape,
        vmap_method="sequential",
        input_layouts=([0, 1], [0, 1, 2], [0], [0]),
        output_layouts=[0, 1],
    )
    group_sizes_i32 = group_sizes if group_sizes.dtype == jnp.int32 else group_sizes.astype(jnp.int32)
    group_offset_i32 = group_offset if group_offset.dtype == jnp.int32 else group_offset.astype(jnp.int32)
    return call(lhs, rhs, group_sizes_i32, group_offset_i32)


def _ragged_dot_group_offset_jax(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
) -> jax.Array:
    offset = group_offset[0]
    m = lhs.shape[0]
    g_local = rhs.shape[0]

    assert g_local > 0, "rhs must have at least one group"

    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]

    token_idx = jnp.arange(m)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)

    local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
    adjusted_group_sizes = local_group_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)

    result = lax.ragged_dot(
        lhs,
        rhs,
        adjusted_group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    return jnp.where(valid_mask[:, None], result, 0)


def _ragged_dot_group_offset_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
) -> jax.Array:
    cublas_out = _ragged_dot_cublas(lhs, rhs, group_sizes, group_offset, preferred_element_type)
    if cublas_out is not None:
        return cublas_out
    return _ragged_dot_group_offset_jax(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


@jax.custom_vjp
def _ragged_dot_group_offset(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
) -> jax.Array:
    return _ragged_dot_group_offset_impl(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


def _ragged_dot_group_offset_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    precision=None,
    preferred_element_type=None,
):
    out = _ragged_dot_group_offset_impl(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    return out, (lhs, rhs, group_sizes, group_offset, precision, preferred_element_type)


def _ragged_dot_group_offset_bwd(res, g):
    lhs, rhs, group_sizes, group_offset, precision, preferred_element_type = res

    def _pure(lhs_, rhs_):
        return _ragged_dot_group_offset_jax(
            lhs_,
            rhs_,
            group_sizes,
            group_offset,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    _, vjp_fn = jax.vjp(_pure, lhs, rhs)
    dlhs, drhs = vjp_fn(g)
    return dlhs, drhs, None, None, None, None


_ragged_dot_group_offset.defvjp(_ragged_dot_group_offset_fwd, _ragged_dot_group_offset_bwd)


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
    group_offset = _normalize_group_offset(group_offset)
    if group_offset is None:
        return lax.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    g_local = rhs.shape[0]
    assert g_local > 0, "rhs must have at least one group"

    return _ragged_dot_group_offset(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


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

    @jax.shard_map(
        mesh=get_abstract_mesh(),
        in_specs=in_specs,
        out_specs=PartitionSpec(),
        axis_names={"ep"},
        check_vma=False,
    )
    def _body(state, *fn_args):
        module_shard = nnx.merge(graphdef, state)
        return func(module_shard, *fn_args)

    return _body(state, *args)
