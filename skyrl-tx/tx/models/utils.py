"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights using jax.vmap
- forward_layers: Unified forward pass using scan (skips KV cache during training)

Prerequisites:
- Layers must be created with create_stacked_layers (stacked weights with proper sharding)
- KVCache must use stacked format: (num_layers, batch, seq, heads, dim)
"""

from typing import Callable

from flax import nnx
import jax
import jax.numpy as jnp

from tx.utils.generator import KVCache


def create_stacked_layers(
    create_layer_fn: Callable[[nnx.Rngs], nnx.Module],
    num_layers: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Create stacked decoder layers with proper sharding propagation.

    This creates a single module object where all parameters have shape (num_layers, ...).
    This enables efficient scanning over layers without runtime stacking.

    Uses jax.vmap on state initialization to properly propagate sharding:
    1. Creates a single layer to get the graphdef structure
    2. Uses jax.vmap to initialize stacked state with different random keys
    3. Sharding is naturally propagated through jax.vmap

    Args:
        create_layer_fn: Function that takes rngs and returns a single layer module.
        num_layers: Number of layers to create.
        rngs: Random number generators for initialization.

    Returns:
        A single module with stacked parameters and correct sharding.

    Example:
        >>> def create_layer(rngs):
        ...     return Llama3DecoderLayer(config, dtype=dtype, rngs=rngs)
        >>> layers = create_stacked_layers(create_layer, config.num_hidden_layers, rngs)
        >>> # layers.self_attn.q_proj.kernel.shape == (num_layers, hidden, head_dim*num_heads)
    """
    # Create the first layer to get the graphdef structure
    first_layer = create_layer_fn(rngs)
    graphdef, _ = nnx.split(first_layer)

    # Split random keys for each layer
    keys = jax.random.split(rngs.params(), num_layers)

    # Use jax.vmap to create stacked state - this naturally propagates sharding
    def init_layer_state(key):
        layer = create_layer_fn(nnx.Rngs(key))
        _, state = nnx.split(layer)
        return state

    stacked_state = jax.vmap(init_layer_state)(keys)

    # Update sharding_names metadata to add leading None for the layer dimension
    # jax.vmap stacks arrays correctly but doesn't update the metadata
    def update_sharding_names(var):
        if isinstance(var, nnx.Variable) and hasattr(var, 'sharding_names') and var.sharding_names is not None:
            return var.replace(sharding_names=(None,) + tuple(var.sharding_names))
        return var

    stacked_state = jax.tree.map(
        update_sharding_names, stacked_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
    )

    # Merge back into a module with the shared graphdef
    return nnx.merge(graphdef, stacked_state)


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
    """Unified forward pass through stacked decoder layers using scan.

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
    is_decode = kv_cache is not None

    if is_decode:
        # --- DECODING PATH (Carry Update) ---
        # Pass cache as carry to allow in-place updates via .at[i].set()
        # This avoids copying the entire KV cache at each layer.

        # Add layer indices to xs so we know which slice to update
        layer_ids = jnp.arange(num_layers)
        xs = (layer_state, layer_ids)

        # Carry holds (hidden_states, keys, values)
        init_carry = (hidden_states, kv_cache.keys, kv_cache.values)

        def decode_body_fn(carry, xs_i):
            hs, keys, values = carry
            layer_params, i = xs_i

            # Reconstruct layer
            layer = nnx.merge(layer_graphdef, layer_params)

            # Read current layer's cache (Dynamic Slice)
            layer_k = keys[i]
            layer_v = values[i]

            # Forward pass
            new_hs, (k, v) = layer(
                hs,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=(layer_k, layer_v),
            )

            # In-place update of the global cache (Dynamic Update Slice)
            # XLA fuses this into a mutable write
            new_keys = keys.at[i].set(k)
            new_values = values.at[i].set(v)

            hs_output = new_hs if output_hidden_states else None

            # Return updated carry, and any sequence outputs (hidden states)
            return (new_hs, new_keys, new_values), hs_output

        # Run scan
        (final_hs, final_keys, final_values), all_hs = jax.lax.scan(
            decode_body_fn, init_carry, xs, unroll=8
        )

        # Reconstruct KVCache from the updated carry arrays
        new_kv_cache = KVCache(
            keys=final_keys,
            values=final_values,
            cache_position=kv_cache.cache_position + positions.shape[1],
        )

        # Handle hidden states output
        all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

        return final_hs, all_hidden_states, new_kv_cache

    else:
        # --- PREFILL / TRAINING PATH (Sequence Stack) ---
        # Standard scan where we collect outputs into new arrays

        def prefill_body_fn(hs, layer_params):
            layer = nnx.merge(layer_graphdef, layer_params)
            new_hs, (k, v) = layer(
                hs,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=None,
            )
            hs_output = new_hs if output_hidden_states else None

            if is_training:
                k = v = None

            return new_hs, (hs_output, k, v)

        if gradient_checkpointing:
            prefill_body_fn = jax.checkpoint(prefill_body_fn)

        final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(
            prefill_body_fn, hidden_states, layer_state, unroll=8
        )

        # [embed, layer0_out, ..., layer(N-2)_out]; final layer output gets normed by caller
        all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

        if is_training:
            new_kv_cache = None
        else:
            # Prefill mode: build cache from collected k,v outputs
            new_kv_cache = KVCache.from_layer_outputs(all_keys, all_values, attention_mask)

        return final_hs, all_hidden_states, new_kv_cache
