from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, PartitionSpec, get_abstract_mesh

from tx.layers.lora import LoRAEmbed, LoRAExpert, LoRALinear
from tx.layers.util import prepare_routing
from tx.layers.rotary_embedding import apply_rope
from tx.models.configs import Qwen3Config
from tx.layers.layernorm import RMSNorm
from tx.models.types import CausalLMOutput, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache, compute_positions


class Qwen3Attention(nnx.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        tp = get_abstract_mesh().shape.get("tp", 1)
        shard_attention_heads = config.shard_attention_heads
        if shard_attention_heads:
            assert self.num_heads % tp == 0, f"num_heads={self.num_heads} must be divisible by tp={tp}"
            assert self.num_kv_heads % tp == 0, f"num_kv_heads={self.num_kv_heads} must be divisible by tp={tp}"
        tp_shard = "tp" if shard_attention_heads else None
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads

        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
            rngs=rngs,
        )
        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
            rngs=rngs,
        )
        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
            rngs=rngs,
        )
        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (tp_shard, "fsdp")),
            rngs=rngs,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape

        # Project and reshape to [B, T, num_heads, head_dim]
        q = self.q_norm(self.q_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, positions, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, positions, self.head_dim, self.config.rope_theta)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache, cache_position = kv_cache
            k = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_position, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_position, 0, 0))

        updated_cache = (k, v)

        # Attention (causal only during prefill, GQA handled natively by dot_product_attention)
        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / self.head_dim**0.5,
            mask=attention_mask[:, None, None, :].astype(bool),
            is_causal=kv_cache is None,
        )

        output = attn_output.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(output, adapter_indices=adapter_indices), updated_cache


class Qwen3MLP(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("tp", "fsdp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(nnx.silu(gate_out) * up_out, adapter_indices)


class Qwen3Experts(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        mesh = get_abstract_mesh()
        ep_axis = "ep" if "ep" in mesh.axis_names else None
        fwd_spec = (ep_axis, "fsdp", "tp") if ep_axis is not None else (None, "fsdp", "tp")
        bwd_spec = (ep_axis, "tp", "fsdp") if ep_axis is not None else (None, "tp", "fsdp")
        self.gate_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), fwd_spec),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), fwd_spec),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), bwd_spec),
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: jax.Array, router_logits: jax.Array, adapter_indices: jax.Array | None = None
    ) -> jax.Array:
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = nnx.softmax(routing_weights, axis=-1)
        mesh = get_abstract_mesh()
        ep_size = mesh.shape.get("ep", 1)
        if ep_size == 1:
            return self._run_single_host(hidden_states, routing_weights, selected_experts, adapter_indices)
        return self._run_expert_parallel(hidden_states, routing_weights, selected_experts, adapter_indices, ep_size)

    def _run_single_host(
        self,
        hidden_states: jax.Array,
        routing_weights: jax.Array,
        selected_experts: jax.Array,
        adapter_indices: jax.Array | None,
    ) -> jax.Array:
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, self.config.num_experts_per_tok, axis=0)
        adapter_indices_expanded = (
            jnp.repeat(adapter_indices, self.config.num_experts_per_tok) if adapter_indices is not None else None
        )
        hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
            hidden_states_expanded,
            selected_experts_flat,
            self.config.num_experts,
            adapter_indices=adapter_indices_expanded,
        )
        gate_out = self.gate_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        up_out = self.up_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        down_out = self.down_proj(nnx.silu(gate_out) * up_out, group_sizes, adapter_indices_sorted)
        unsorted_out = down_out[unsort_indices]
        reshaped_out = unsorted_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
        return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)

    def _run_expert_parallel(
        self,
        hidden_states: jax.Array,
        routing_weights: jax.Array,
        selected_experts: jax.Array,
        adapter_indices: jax.Array | None,
        ep_size: int,
    ) -> jax.Array:
        experts_per_axis = self.config.num_experts // ep_size
        if self.config.num_experts % ep_size != 0:
            raise ValueError("Number of experts must be divisible by ep axis size")
        total_pairs = hidden_states.shape[0] * self.config.num_experts_per_tok
        capacity = getattr(self.config, "expert_capacity", None)
        if capacity is None:
            capacity = total_pairs
        adapter_arg = adapter_indices if adapter_indices is not None else jnp.zeros((hidden_states.shape[0],), jnp.int32)

        # Split the expert modules to extract state for shard_map
        gate_graphdef, gate_state = nnx.split(self.gate_proj)
        up_graphdef, up_state = nnx.split(self.up_proj)
        down_graphdef, down_state = nnx.split(self.down_proj)
        max_lora_adapters = self.gate_proj.max_lora_adapters
        max_lora_rank = self.gate_proj.max_lora_rank

        def segment_starts(lengths: jax.Array) -> jax.Array:
            zeros = jnp.zeros((1,), dtype=jnp.int32)
            if lengths.size == 0:
                return zeros
            return jnp.concatenate([zeros, jnp.cumsum(lengths[:-1], dtype=jnp.int32)])

        def expert_parallel_fn(
            tokens: jax.Array,
            weights: jax.Array,
            experts: jax.Array,
            adapters: jax.Array,
            gate_s: nnx.State,
            up_s: nnx.State,
            down_s: nnx.State,
        ) -> jax.Array:
            # Merge states back into modules
            gate_proj = nnx.merge(gate_graphdef, gate_s)
            up_proj = nnx.merge(up_graphdef, up_s)
            down_proj = nnx.merge(down_graphdef, down_s)

            num_tokens = tokens.shape[0]
            hidden_size = tokens.shape[-1]
            expanded_tokens = jnp.repeat(tokens, self.config.num_experts_per_tok, axis=0)
            expanded_experts = experts.reshape(-1)
            expanded_indices = jnp.arange(expanded_tokens.shape[0], dtype=jnp.int32)
            adapter_flat = jnp.repeat(adapters, self.config.num_experts_per_tok, axis=0)
            target_ep = expanded_experts // experts_per_axis
            local_expert = expanded_experts % experts_per_axis
            sort_perm = jnp.argsort(target_ep, stable=True)
            target_ep_sorted = target_ep[sort_perm]
            tokens_sorted = expanded_tokens[sort_perm]
            local_expert_sorted = local_expert[sort_perm]
            dispatch_sorted = expanded_indices[sort_perm]
            origin_sorted = jnp.full_like(target_ep_sorted, jax.lax.axis_index("ep"), dtype=jnp.int32)
            adapters_sorted = adapter_flat[sort_perm]

            send_sizes = jnp.bincount(target_ep_sorted, length=ep_size).astype(jnp.int32)
            input_offsets = segment_starts(send_sizes)
            recv_sizes = jax.lax.all_to_all(send_sizes, "ep", 0, 0, tiled=True)
            recv_offsets = segment_starts(recv_sizes)
            output_offsets = jax.lax.all_to_all(recv_offsets, "ep", 0, 0, tiled=True)
            token_buffer = jnp.zeros((capacity, hidden_size), dtype=tokens.dtype)
            expert_buffer = jnp.zeros((capacity,), dtype=jnp.int32)
            dispatch_buffer = jnp.zeros((capacity,), dtype=jnp.int32)
            origin_buffer = jnp.zeros((capacity,), dtype=jnp.int32)
            adapter_buffer = jnp.zeros((capacity,), dtype=adapters.dtype)
            tokens_recv = jax.lax.ragged_all_to_all(
                tokens_sorted,
                token_buffer,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name="ep",
            )
            expert_recv = jax.lax.ragged_all_to_all(
                local_expert_sorted,
                expert_buffer,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name="ep",
            )
            dispatch_recv = jax.lax.ragged_all_to_all(
                dispatch_sorted,
                dispatch_buffer,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name="ep",
            )
            origin_recv = jax.lax.ragged_all_to_all(
                origin_sorted,
                origin_buffer,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name="ep",
            )
            adapters_recv = jax.lax.ragged_all_to_all(
                adapters_sorted,
                adapter_buffer,
                input_offsets,
                send_sizes,
                output_offsets,
                recv_sizes,
                axis_name="ep",
            )

            total_recv = jnp.sum(recv_sizes, dtype=jnp.int32)
            # Don't slice - use full buffers. Mark invalid entries with origin=ep_size
            # so they sort to the end and aren't counted in bincount(length=ep_size)
            valid_recv_mask = jnp.arange(capacity) < total_recv
            tokens_local = tokens_recv
            expert_local = expert_recv
            dispatch_local = dispatch_recv
            origin_local = jnp.where(valid_recv_mask, origin_recv, ep_size)
            adapters_local = adapters_recv

            routed_tokens, group_sizes, unsort_idx, adapters_grouped = prepare_routing(
                tokens_local,
                expert_local,
                experts_per_axis,
                adapter_indices=adapters_local,
            )

            # Apply expert computations using merged modules
            gate_out = gate_proj(routed_tokens, group_sizes, adapters_grouped)
            up_out = up_proj(routed_tokens, group_sizes, adapters_grouped)
            down_out = down_proj(nnx.silu(gate_out) * up_out, group_sizes, adapters_grouped)
            expert_outputs = down_out[unsort_idx]

            back_perm = jnp.argsort(origin_local, stable=True)
            origin_sorted_back = origin_local[back_perm]
            outputs_sorted_back = expert_outputs[back_perm]
            dispatch_sorted_back = dispatch_local[back_perm]
            send_sizes_back = jnp.bincount(origin_sorted_back, length=ep_size).astype(jnp.int32)
            input_offsets_back = segment_starts(send_sizes_back)
            recv_sizes_back = jax.lax.all_to_all(send_sizes_back, "ep", 0, 0, tiled=True)
            recv_offsets_back = segment_starts(recv_sizes_back)
            output_offsets_back = jax.lax.all_to_all(recv_offsets_back, "ep", 0, 0, tiled=True)
            output_buffer = jnp.zeros((total_pairs, hidden_size), dtype=expert_outputs.dtype)
            index_buffer = jnp.zeros((total_pairs,), dtype=jnp.int32)
            outputs_returned = jax.lax.ragged_all_to_all(
                outputs_sorted_back,
                output_buffer,
                input_offsets_back,
                send_sizes_back,
                output_offsets_back,
                recv_sizes_back,
                axis_name="ep",
            )
            indices_returned = jax.lax.ragged_all_to_all(
                dispatch_sorted_back,
                index_buffer,
                input_offsets_back,
                send_sizes_back,
                output_offsets_back,
                recv_sizes_back,
                axis_name="ep",
            )
            # Don't slice - reverse so valid entries (at the beginning) scatter last
            # and overwrite any invalid entries that scattered to the same position
            scatter_buffer = jnp.zeros((total_pairs, hidden_size), dtype=expert_outputs.dtype)
            scatter_buffer = scatter_buffer.at[indices_returned[::-1]].set(outputs_returned[::-1])
            reshaped = scatter_buffer.reshape(num_tokens, self.config.num_experts_per_tok, hidden_size)
            return jnp.sum(reshaped * weights[..., None], axis=1)

        def filter_out_ep(spec_tree):
            """Filter partition specs to remove 'ep' axis (handled by shard_map's axis_names)."""
            def filter_spec(spec):
                if spec is None or spec == P():
                    return P()
                new_axes = []
                for axis in spec:
                    if isinstance(axis, tuple):
                        # Multi-axis sharding: remove 'ep' if present, keep others
                        filtered = tuple(a for a in axis if a != "ep")
                        new_axes.append(filtered or None)
                    else:
                        # Single axis: remove if it's 'ep', keep otherwise
                        new_axes.append(None if axis == "ep" else axis)
                return P(*new_axes)
            return jax.tree.map(filter_spec, spec_tree, is_leaf=lambda x: isinstance(x, P))

        # Get partition specs from the states, removing 'ep' (handled by shard_map)
        gate_state_specs = filter_out_ep(nnx.get_partition_spec(gate_state))
        up_state_specs = filter_out_ep(nnx.get_partition_spec(up_state))
        down_state_specs = filter_out_ep(nnx.get_partition_spec(down_state))

        in_specs = (
            P(), P(), P(), P(),  # tokens, weights, experts, adapters
            gate_state_specs,
            up_state_specs,
            down_state_specs,
        )

        sharded_fn = jax.shard_map(
            expert_parallel_fn,
            mesh=get_abstract_mesh(),
            in_specs=in_specs,
            out_specs=P(),
            axis_names={"ep",},
        )
        return sharded_fn(
            hidden_states, routing_weights, selected_experts, adapter_arg,
            gate_state, up_state, down_state,
        )


class Qwen3MoeSparseMoeBlock(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None)),
            rngs=rngs,
        )
        self.experts = Qwen3Experts(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        adapter_indices: jax.Array | None = None,
        return_router_logits: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        (batch_size, seq_len, hidden_size) = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        # Expand adapter_indices to match flattened hidden_states
        if adapter_indices is not None:
            adapter_indices = jnp.repeat(adapter_indices, seq_len)
        router_logits = self.gate(hidden_states)

        hidden_states = self.experts(hidden_states, router_logits, adapter_indices)
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)

        if return_router_logits:
            return hidden_states, router_logits
        return hidden_states


class Qwen3DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Qwen3Attention(config, dtype=dtype, rngs=rngs)
        if getattr(config, "num_experts", None):
            self.mlp = Qwen3MoeSparseMoeBlock(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = Qwen3MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + mlp_output

        return hidden_states, updated_cache


class Qwen3Model(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config

        self.embed_tokens = LoRAEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(), ("tp", None)),
            rngs=rngs,
        )
        self.layers = nnx.List(
            [Qwen3DecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embed_tokens(input_ids, adapter_indices=adapter_indices)
        all_hidden_states: list[jax.Array] = []
        updated_keys, updated_values = [], []

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, (k, v) = layer(
                hidden_states,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=kv_cache and (kv_cache.keys[layer_idx], kv_cache.values[layer_idx], kv_cache.cache_position),
            )
            updated_keys.append(k)
            updated_values.append(v)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Increment cache_position if cache exists, or use sequence length for new cache
        new_cache_position = kv_cache.cache_position + 1 if kv_cache is not None else input_ids.shape[1]

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=KVCache(keys=updated_keys, values=updated_values, cache_position=new_cache_position),
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Qwen3ForCausalLM(nnx.Module, GeneratorMixin):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs)
        if not self.config.tie_word_embeddings:
            self.lm_head = LoRALinear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp")),
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                rngs=rngs,
            )

    @staticmethod
    def is_lora_param(path: tuple, _value) -> bool:
        """Return True if a parameter path corresponds to LoRA weights."""
        return any(name in path for name in ("lora_A", "lora_B"))

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
        kv_cache: KVCache | None = None,
    ) -> CausalLMOutput:
        if positions is None:
            positions = compute_positions(attention_mask)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            positions=positions,
            output_hidden_states=output_hidden_states,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
        )
        hidden_states = outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.embedding.value.T
        else:
            logits = self.lm_head(hidden_states, adapter_indices=adapter_indices)

        return CausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )


Qwen3MoeForCausalLM = Qwen3ForCausalLM
