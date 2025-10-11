import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from tx.layers.lora import LoRAExpert, LoRALinear
from tx.layers.util import Param, prepare_routing


class RMSNorm(nn.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


def apply_rope(inputs: torch.Tensor, position_ids: torch.Tensor, head_dim: int, theta: int) -> torch.Tensor:
    fraction = 2 * torch.arange(0, head_dim // 2, dtype=torch.float32, device=inputs.device) / head_dim
    timescale = theta ** fraction
    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = torch.sin(x), torch.cos(x)
    a, b = torch.chunk(inputs, 2, dim=-1)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(inputs.dtype)


class Qwen3Attention(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)

        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            device=device,
        )
        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            device=device,
        )
        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            device=device,
        )
        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            device=device,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        adapter_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Reshape each: [B,T,H*D] -> [B,T,H,D]
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)

        position_ids = torch.arange(x.shape[1], device=x.device)[None, :].repeat(x.shape[0], 1)

        q = apply_rope(q, position_ids, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, position_ids, self.head_dim, self.config.rope_theta)

        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_groups, dim=2)
            v = v.repeat_interleave(num_groups, dim=2)

        # Use scaled_dot_product_attention
        # Shape: [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle attention masking
        # For padding: attention_mask is 1 for valid tokens, 0 for padding
        # In attention matrix: rows are queries, columns are keys
        # We mask KEY positions (columns) that are padding
        if attention_mask is not None:
            # Create 4D attention mask: [B, 1, T_query, T_key]
            # attention_mask shape: [B, T] where 1 = valid, 0 = padding
            # Expand to [B, 1, 1, T] to broadcast over queries
            key_padding_mask = attention_mask[:, None, None, :].bool()  # [B, 1, 1, T]
            # Create causal mask: [T, T] where False = can attend, True = cannot attend
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            # Combine: can attend if key is not padding AND position is allowed by causal mask
            # True = can attend, False = mask out
            attn_mask = key_padding_mask & ~causal_mask[None, None, :, :]  # [B, 1, T, T]
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),  # Use is_causal only when no padding mask
        )

        # [B, H, T, D] -> [B, T, H, D]
        attn_output = attn_output.transpose(1, 2)
        attn_out_flat = attn_output.reshape(B, T, self.num_heads * self.head_dim)  # [B,T,H,D] -> [B,T,H*D]
        return self.o_proj(attn_out_flat, adapter_indices=adapter_indices)


class Qwen3MLP(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            device=device,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            device=device,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            device=device,
        )

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(F.silu(gate_out) * up_out, adapter_indices)


class Qwen3Experts(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)

        self.gate_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            device=device,
        )
        self.up_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            device=device,
        )
        self.down_proj = LoRAExpert(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            device=device,
        )

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, adapter_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Get top-k experts for each token and compute routing weights
        routing_weights, selected_experts = torch.topk(router_logits, k=self.config.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Prepare for grouped_mm by sorting tokens based on their assigned expert
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = hidden_states.repeat_interleave(self.config.num_experts_per_tok, dim=0)
        adapter_indices_expanded = (
            adapter_indices.repeat_interleave(self.config.num_experts_per_tok) if adapter_indices is not None else None
        )
        hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
            hidden_states_expanded,
            selected_experts_flat,
            self.config.num_experts,
            adapter_indices=adapter_indices_expanded,
        )

        # Apply expert layers using LoRAExpert
        gate_out = self.gate_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        up_out = self.up_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        down_out = self.down_proj(F.silu(gate_out) * up_out, group_sizes, adapter_indices_sorted)

        # Unsort and combine the expert outputs
        unsorted_out = down_out[unsort_indices]
        reshaped_out = unsorted_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
        return torch.sum(reshaped_out * routing_weights[..., None], dim=1)


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.experts = Qwen3Experts(config, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        adapter_indices: torch.Tensor | None = None,
        return_router_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        (batch_size, seq_len, hidden_size) = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        # Expand adapter_indices to match flattened hidden_states
        if adapter_indices is not None:
            adapter_indices = adapter_indices.repeat_interleave(seq_len)
        router_logits = self.gate(hidden_states)

        hidden_states = self.experts(hidden_states, router_logits, adapter_indices)
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)

        if return_router_logits:
            return hidden_states, router_logits
        return hidden_states


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.self_attn = Qwen3Attention(config, dtype=dtype, device=device)
        if getattr(config, "num_experts", None):
            self.mlp = Qwen3MoeSparseMoeBlock(config, dtype=dtype, device=device)
        else:
            self.mlp = Qwen3MLP(config, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        adapter_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            adapter_indices=adapter_indices,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states: list[torch.Tensor] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                adapter_indices=adapter_indices,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


class Qwen3ForCausalLM(nn.Module):

    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, device=device)
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=dtype,
                device=device,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            adapter_indices=adapter_indices,
        )
        hidden_states = outputs["last_hidden_state"]
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.weight.T
        else:
            logits = self.lm_head(hidden_states)

        return {"logits": logits, **outputs}
