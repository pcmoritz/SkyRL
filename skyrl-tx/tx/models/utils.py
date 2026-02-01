"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights using nnx.vmap
- forward_layers: Unified forward pass using scan for prefill, Python loop for decode

Prerequisites:
- Layers must be created with nnx.vmap (stacked weights)
- KVCache uses unstacked format: list of per-layer caches
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

    Uses scan for prefill/training (benefits from gradient checkpointing).
    Uses Python loop for decode (better memory efficiency with unstacked cache).

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

    # For decode mode, use Python loop with unstacked cache (mimics original fast impl)
    if is_decode:
        return _forward_layers_decode_loop(
            layer_graphdef,
            layer_state,
            hidden_states,
            num_layers,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
        )

    # Prefill/training mode: use scan
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

        if is_training:
            k = v = None

        return new_hs, (hs_output, k, v)

    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, layer_state)

    # [embed, layer0_out, ..., layer(N-2)_out]; final layer output gets normed by caller
    all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

    if is_training:
        new_kv_cache = None
    else:
        # Prefill mode: convert stacked outputs to list for unstacked KVCache
        new_kv_cache = KVCache.from_layer_outputs(
            list(all_keys), list(all_values), attention_mask
        )

    return final_hs, all_hidden_states, new_kv_cache


def _forward_layers_decode_loop(
    layer_graphdef: nnx.GraphDef,
    layer_state: nnx.State,
    hidden_states: jax.Array,
    num_layers: int,
    *,
    attention_mask: jax.Array,
    positions: jax.Array,
    adapter_indices: jax.Array | None,
    kv_cache: KVCache,
    output_hidden_states: bool,
) -> tuple[jax.Array, list[jax.Array], KVCache]:
    """Decode using Python loop with unstacked KV cache.

    This mimics the original fast implementation where each layer has its own
    separate cache tensor. Updates to one layer's cache don't affect others.
    """
    hs = hidden_states
    all_hidden_states = [hidden_states] if output_hidden_states else []

    # Work with the unstacked cache lists directly
    new_keys = list(kv_cache.keys)
    new_values = list(kv_cache.values)

    for i in range(num_layers):
        # Get layer i's parameters
        layer_params_i = jax.tree.map(lambda x, idx=i: x[idx], layer_state)
        layer = nnx.merge(layer_graphdef, layer_params_i)

        # Use this layer's separate cache tensor
        layer_kv = (new_keys[i], new_values[i])

        # Forward through layer - layer returns updated cache
        hs, (k, v) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )

        # Update this layer's cache (independent tensor, no stacking overhead)
        new_keys[i] = k
        new_values[i] = v

        if output_hidden_states and i < num_layers - 1:
            all_hidden_states.append(hs)

    new_kv_cache = KVCache(
        keys=new_keys,
        values=new_values,
        cache_position=kv_cache.cache_position + positions.shape[1],
    )

    return hs, all_hidden_states, new_kv_cache
