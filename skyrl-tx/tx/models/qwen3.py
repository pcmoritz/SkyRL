from functools import partial

from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

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
    """MoE experts with expert parallelism support.

    Expert weights are sharded along the "ep" mesh axis. When ep=1 in the mesh,
    sharding on "ep" is a no-op, so this works for both EP and non-EP cases.
    """

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("ep", "fsdp", "tp")),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("ep", "fsdp", "tp")),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("ep", "tp", "fsdp")),
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: jax.Array, router_logits: jax.Array, adapter_indices: jax.Array | None = None
    ) -> jax.Array:
        # Get top-k experts for each token and compute routing weights
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = nnx.softmax(routing_weights, axis=-1)

        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, self.config.num_experts_per_tok, axis=0)
        adapter_indices_expanded = (
            jnp.repeat(adapter_indices, self.config.num_experts_per_tok) if adapter_indices is not None else None
        )
        mesh = get_abstract_mesh()
        ep_size = mesh.shape.get("ep", 1) if mesh else 1

        if ep_size > 1:
            expert_out = self._forward_ep(
                hidden_states_expanded, selected_experts_flat, adapter_indices_expanded, mesh, ep_size
            )
        else:
            hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
                hidden_states_expanded, selected_experts_flat, self.config.num_experts, adapter_indices=adapter_indices_expanded
            )
            gate_out = self.gate_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
            up_out = self.up_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
            down_out = self.down_proj(nnx.silu(gate_out) * up_out, group_sizes, adapter_indices_sorted)
            expert_out = down_out[unsort_indices]

        reshaped_out = expert_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
        return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)

    def _forward_ep(self, hidden_states, selected_experts, adapter_indices, mesh, ep_size):
        """EP forward using shard_map with nnx.split/merge for clean state handling."""
        num_tokens, H = hidden_states.shape
        experts_per_device = self.config.num_experts // ep_size

        # Map global expert -> (target_device, local_expert)
        target_device = selected_experts // experts_per_device
        local_expert = selected_experts % experts_per_device

        # Sort to group tokens by target device
        sort_idx = jnp.argsort(target_device)

        # Pad num_tokens to be divisible by ep_size for shard_map
        pad_size = (ep_size - (num_tokens % ep_size)) % ep_size
        padded_size = num_tokens + pad_size

        # Extend sort_idx to cover padded elements (padding goes to last device)
        if pad_size > 0:
            sort_idx = jnp.concatenate([sort_idx, jnp.arange(num_tokens, padded_size)])

        def pad_and_permute(x, default=0):
            if pad_size > 0:
                x = jnp.pad(x, ((0, pad_size), (0, 0)), constant_values=default)
            return x[sort_idx].reshape(ep_size, -1, x.shape[-1])

        # Prepare inputs for shard_map
        x_sharded = pad_and_permute(hidden_states[:, None] if hidden_states.ndim == 1 else hidden_states)
        x_sharded = x_sharded.squeeze(-1) if hidden_states.ndim == 1 else x_sharded
        adapters_sharded = pad_and_permute(adapter_indices[:, None]).squeeze(-1) if adapter_indices is not None else None
        local_expert_sharded = pad_and_permute(local_expert[:, None]).squeeze(-1)

        # Extract weights manually - weight has ep on axis 0, lora has ep on axis 1
        gw, uw, dw = self.gate_proj.weight.value, self.up_proj.weight.value, self.down_proj.weight.value
        gA, gB, gs = self.gate_proj.lora_A.value, self.gate_proj.lora_B.value, self.gate_proj.lora_scaling.value
        uA, uB, us = self.up_proj.lora_A.value, self.up_proj.lora_B.value, self.up_proj.lora_scaling.value
        dA, dB, ds = self.down_proj.lora_A.value, self.down_proj.lora_B.value, self.down_proj.lora_scaling.value

        W = P("ep", None, None)  # weight: (experts, in, out)
        L = P(None, "ep", None, None)  # lora: (adapters, experts, ...)
        S = P(None)  # scaling: (adapters,)

        @partial(shard_map, mesh=mesh,
                 in_specs=(P("ep", None, None), P("ep", None), P("ep", None),
                           W, W, W, L, L, S, L, L, S, L, L, S),
                 out_specs=P("ep", None, None), check_rep=False)
        def ep_step(x, l_expert, adapters, gw, uw, dw, gA, gB, gs, uA, uB, us, dA, dB, ds):
            # Flatten inputs (shard_map keeps trailing dims)
            l_expert = l_expert.ravel()
            adapters = adapters.ravel() if adapters is not None else None

            # Local routing: sort by local_expert for ragged_dot
            local_sort = jnp.argsort(l_expert)
            x_sorted = x[local_sort]
            adapter_sorted = adapters[local_sort] if adapters is not None else None
            group_sizes = jnp.bincount(l_expert, minlength=experts_per_device, length=experts_per_device)

            # MLP with inline LoRA
            def apply_expert(x, w, lA, lB, ls, adapters):
                base = jax.lax.ragged_dot(x, w, group_sizes)
                # LoRA
                exp_idx = jnp.repeat(jnp.arange(experts_per_device), group_sizes, total_repeat_length=x.shape[0])
                flat_idx = adapters * experts_per_device + exp_idx
                num_groups = lA.shape[0] * experts_per_device
                lA_flat = lA.reshape(num_groups, lA.shape[-2], lA.shape[-1])
                lB_flat = lB.reshape(num_groups, lB.shape[-2], lB.shape[-1])
                x_s, sizes, unsort, _ = prepare_routing(x, flat_idx, num_groups)
                lora = jax.lax.ragged_dot(jax.lax.ragged_dot(x_s, lA_flat, sizes), lB_flat, sizes)
                return base + lora[unsort] * ls[adapters, None]

            g = apply_expert(x_sorted, gw, gA, gB, gs, adapter_sorted)
            u = apply_expert(x_sorted, uw, uA, uB, us, adapter_sorted)
            d = apply_expert(nnx.silu(g) * u, dw, dA, dB, ds, adapter_sorted)
            return d[jnp.argsort(local_sort)]

        # Run EP computation
        output_sharded = ep_step(x_sharded, local_expert_sharded, adapters_sharded,
                                 gw, uw, dw, gA, gB, gs, uA, uB, us, dA, dB, ds)

        # Restore global order
        output_flat = output_sharded.reshape(-1, H)
        if pad_size > 0:
            output_flat = output_flat[:-pad_size]

        # Inverse permutation to restore original token order
        return output_flat[jnp.argsort(sort_idx[:num_tokens])]


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
