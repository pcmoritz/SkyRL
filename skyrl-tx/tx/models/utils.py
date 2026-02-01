"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights using nnx.vmap
- forward_layers: Unified forward pass (scan for training, loop for inference)

Prerequisites:
- Layers must be created with nnx.vmap (stacked weights)
- KVCache uses list format: list of (batch, seq, heads, dim) arrays per layer
"""

from typing import Callable

from flax import nnx
import jax

from tx.utils.generator import KVCache


def create_stacked_layers(
    create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
    num_layers: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create stacked decoder layers using nnx.vmap.

    This creates a single module object where all parameters have shape (num_layers, ...).
    This enables efficient scanning over layers without runtime stacking.

    Args:
        create_layer_fn: Function that takes rngs and returns a single layer module.
        num_layers: Number of layers to create.
        rngs: Random number generators for initialization.

    Returns:
        A single module with stacked parameters.

    Example:
        >>> def create_layer(rngs):
        ...     return Llama3DecoderLayer(config, dtype=dtype, rngs=rngs)
        >>> layers = create_stacked_layers(create_layer, config.num_hidden_layers, rngs)
        >>> # layers.self_attn.q_proj.kernel.shape == (num_layers, hidden, head_dim*num_heads)
    """

    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=(0,), out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
    def vmapped_create(rngs: nnx.Rngs):
        return create_layer_fn(rngs)

    return vmapped_create(rngs)


def forward_layers(
    layers: nnx.Module,
    hidden_states: jax.Array,
    num_layers: int,
    *,
    attention_mask: jax.Array,
    positions: jax.Array,
    adapter_indices: jax.Array | None,
    kv_cache: KVCache | None,
    output_hidden_states: bool,
    gradient_checkpointing: bool,
    is_training: bool = False,
) -> tuple[jax.Array, list[jax.Array], KVCache | None]:
    """Unified forward pass through stacked decoder layers.

    Uses scan for training (efficient, no KV cache needed) and a Python loop
    for inference (supports list-based KV cache).

    Args:
        layers: Stacked decoder layers (created with create_stacked_layers/nnx.vmap).
        hidden_states: Input hidden states of shape (batch, seq, hidden).
        num_layers: Number of decoder layers.
        attention_mask: Attention mask of shape (batch, seq).
        positions: Position indices of shape (batch, seq).
        adapter_indices: Optional LoRA adapter indices of shape (batch,).
        kv_cache: Optional KV cache for decode mode (None for prefill).
        output_hidden_states: Whether to return intermediate hidden states.
        gradient_checkpointing: Whether to use gradient checkpointing (training only).
        is_training: Whether in training mode. Skips KV cache to save memory.

    Returns:
        Tuple of (final_hidden_states, all_hidden_states, kv_cache).
        kv_cache is None when is_training=True.
    """
    assert num_layers > 0, "num_layers must be positive"

    layer_graphdef, layer_state = nnx.split(layers)

    if is_training:
        # Training path: use scan for efficiency (no KV cache needed)
        def body_fn(hs, layer_params):
            layer = nnx.merge(layer_graphdef, layer_params)
            new_hs, (k, v) = layer(
                hs,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=None,
            )
            hs_output = new_hs if output_hidden_states else None
            return new_hs, hs_output

        if gradient_checkpointing:
            body_fn = jax.checkpoint(body_fn)

        final_hs, all_hs = jax.lax.scan(body_fn, hidden_states, layer_state)
        all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []
        return final_hs, all_hidden_states, None
    else:
        # Inference path: use Python loop to support list-based KV cache
        all_hidden_states: list[jax.Array] = []
        updated_keys: list[jax.Array] = []
        updated_values: list[jax.Array] = []

        for layer_idx in range(num_layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # Get the layer params for this layer by indexing into stacked state
            layer_params = jax.tree.map(lambda x: x[layer_idx], layer_state)
            layer_kv = (kv_cache.keys[layer_idx], kv_cache.values[layer_idx]) if kv_cache else None

            layer = nnx.merge(layer_graphdef, layer_params)
            hidden_states, (k, v) = layer(
                hidden_states,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=layer_kv,
            )
            updated_keys.append(k)
            updated_values.append(v)

        new_kv_cache = KVCache.update(kv_cache, updated_keys, updated_values, positions, attention_mask)
        return hidden_states, all_hidden_states, new_kv_cache
