from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec


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


def _replicated_spec_like(array: jax.Array) -> PartitionSpec:
    """Return a PartitionSpec that mirrors an array's sharding but omits 'ep'."""
    sharding = getattr(array, "sharding", None)
    if sharding is not None and hasattr(sharding, "spec"):
        partitions: list = []
        for part in sharding.spec:
            if part is None:
                partitions.append(None)
            elif isinstance(part, tuple):
                filtered = tuple(axis for axis in part if axis != "ep")
                partitions.append(filtered or None)
            else:
                partitions.append(None if part == "ep" else part)
        return PartitionSpec(*partitions)
    ndim = array.ndim if hasattr(array, "ndim") else len(array.shape)
    return PartitionSpec(*((None,) * ndim))


def _local_expert_computation(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None = None,
    expert_kwargs: dict[str, object] | None = None,
) -> jax.Array:
    """Run expert computation locally without expert-parallel sharding."""
    selected_experts_flat = selected_experts.reshape(-1)
    hidden_states_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
    adapter_indices_expanded = (
        jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
    )
    hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
        hidden_states_expanded,
        selected_experts_flat,
        num_experts,
        adapter_indices=adapter_indices_expanded,
    )

    expert_kwargs = expert_kwargs or {}
    expert_out = expert_fn(hidden_states_sorted, group_sizes, adapter_indices_sorted, **expert_kwargs)
    unsorted_out = expert_out[unsort_indices]
    reshaped_out = unsorted_out.reshape(-1, num_experts_per_tok, hidden_size)
    return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)


def expert_parallel_dispatch_combine(
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    routing_weights: jax.Array,
    expert_fn,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    adapter_indices: jax.Array | None = None,
) -> jax.Array:
    """Dispatch tokens to experts and combine outputs, optionally across EP shards."""
    mesh = get_abstract_mesh()
    ep_size = mesh.shape.get("ep", 1) if mesh is not None else 1

    if ep_size == 1:
        return _local_expert_computation(
            hidden_states,
            selected_experts,
            routing_weights,
            expert_fn,
            num_experts,
            num_experts_per_tok,
            hidden_size,
            adapter_indices=adapter_indices,
        )

    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by expert_parallel_size ({ep_size})")

    experts_per_rank = num_experts // ep_size

    def _make_shard_body(has_adapter: bool):

        def _shard_body(
            shard_hidden_states: jax.Array,
            shard_selected_experts: jax.Array,
            shard_routing_weights: jax.Array,
            shard_adapter_indices: jax.Array | None = None,
        ) -> jax.Array:
            axis_index = jax.lax.axis_index("ep")
            shard_start = axis_index * experts_per_rank
            shard_end = shard_start + experts_per_rank

            local_mask = (shard_selected_experts >= shard_start) & (shard_selected_experts < shard_end)
            local_routing = shard_routing_weights * local_mask.astype(shard_routing_weights.dtype)
            local_selected = jnp.where(
                local_mask,
                shard_selected_experts - shard_start,
                jnp.zeros_like(shard_selected_experts),
            )

            const_start = jax.lax.stop_gradient(shard_start)
            local_output = _local_expert_computation(
                shard_hidden_states,
                local_selected,
                local_routing,
                expert_fn,
                experts_per_rank,
                num_experts_per_tok,
                hidden_size,
                adapter_indices=shard_adapter_indices if has_adapter else None,
                expert_kwargs={"expert_start": const_start, "num_experts_chunk": experts_per_rank},
            )
            return jax.lax.psum(local_output, axis_name="ep")

        return _shard_body

    axis_names = {"ep"}
    hidden_out_spec = _replicated_spec_like(hidden_states)

    def _build_specs(has_adapter: bool):
        common_specs = (
            _replicated_spec_like(hidden_states),
            _replicated_spec_like(selected_experts),
            _replicated_spec_like(routing_weights),
        )
        if not has_adapter:
            return common_specs, hidden_out_spec
        return common_specs + (_replicated_spec_like(adapter_indices),), hidden_out_spec

    if adapter_indices is None:
        in_specs, out_spec = _build_specs(has_adapter=False)
        sharded_fn = jax.shard_map(
            _make_shard_body(has_adapter=False),
            in_specs=in_specs,
            out_specs=out_spec,
            axis_names=axis_names,
        )
        return sharded_fn(hidden_states, selected_experts, routing_weights)

    in_specs, out_spec = _build_specs(has_adapter=True)
    sharded_fn = jax.shard_map(
        _make_shard_body(has_adapter=True),
        in_specs=in_specs,
        out_specs=out_spec,
        axis_names=axis_names,
    )
    return sharded_fn(hidden_states, selected_experts, routing_weights, adapter_indices)
