"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights using nnx.vmap
- forward_layers: Forward pass using scan for training/prefill and loops for decode

Prerequisites:
- Layers must be created with nnx.vmap (stacked weights)
- KVCache uses list format: list of (batch, seq, heads, dim) arrays per layer
"""

from typing import Callable

from flax import nnx
import jax

from tx.utils.generator import KVCache


def extract_layer_params(layers: nnx.Module, num_layers: int) -> tuple:
    """Pre-extract layer params from stacked layers for efficient decode.

    This extracts all per-layer params once, so they don't need to be
    re-extracted on each decode iteration.

    Args:
        layers: Stacked decoder layers (created with create_stacked_layers/nnx.vmap).
        num_layers: Number of decoder layers.

    Returns:
        Tuple of (layer_graphdef, all_layer_params) where all_layer_params is a list
        of per-layer params that can be passed to forward_layers via pre_extracted_layers.
    """
    layer_graphdef, layer_state = nnx.split(layers)
    all_layer_params = [
        jax.tree.map(lambda x, i=i: x[i], layer_state) for i in range(num_layers)
    ]
    return (layer_graphdef, all_layer_params)


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
    pre_extracted_layers: tuple | None = None,
) -> tuple[jax.Array, list[jax.Array], KVCache | None]:
    """Forward pass through stacked decoder layers.

    Uses jax.lax.scan for training and prefill, Python loops for decode.

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
        pre_extracted_layers: Optional pre-extracted layer params for decode mode.
            Tuple of (layer_graphdef, all_layer_params) where all_layer_params is a list
            of per-layer params. When provided, avoids re-extracting params on each decode step.

    Returns:
        Tuple of (final_hidden_states, all_hidden_states, kv_cache).
        kv_cache is None when is_training=True.
    """
    assert num_layers > 0, "num_layers must be positive"

    is_decode = kv_cache is not None

    # Use pre-extracted layers if provided (for decode), otherwise split here
    if pre_extracted_layers is not None:
        layer_graphdef, all_layer_params = pre_extracted_layers
        layer_state = None  # Not needed when pre-extracted
    else:
        layer_graphdef, layer_state = nnx.split(layers)
        all_layer_params = None

    if is_decode:
        # Decode mode: use Python loop for layer-by-layer processing
        all_hidden_states = [hidden_states] if output_hidden_states else []
        new_keys = []
        new_values = []

        # Use pre-extracted params if available, otherwise extract from layer_state
        if all_layer_params is None:
            all_layer_params = [
                jax.tree.map(lambda x, i=i: x[i], layer_state) for i in range(num_layers)
            ]

        for i in range(num_layers):
            layer = nnx.merge(layer_graphdef, all_layer_params[i])

            # Get the KV cache for this layer
            layer_kv = (kv_cache.keys[i], kv_cache.values[i])

            hidden_states, (k, v) = layer(
                hidden_states,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=layer_kv,
            )

            new_keys.append(k)
            new_values.append(v)

            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(hidden_states)

        new_kv_cache = KVCache(
            keys=new_keys,
            values=new_values,
            cache_position=kv_cache.cache_position + positions.shape[1],
        )

        return hidden_states, all_hidden_states, new_kv_cache

    # Training or prefill mode: use scan
    def body_fn(hs, layer_params):
        # Merge using the sliced params directly - no manual gather needed
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
            # Avoid accumulating large KV tensors for training.
            k = v = None
        return new_hs, (hs_output, k, v)

    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    # Pass layer_state as xs so scan handles the slicing automatically.
    # This avoids capturing layer_state as a closure and manually gathering,
    # which causes slow XLA compilation with jax.checkpoint.
    final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, layer_state)

    # [embed, layer0_out, ..., layer(N-2)_out]; final layer output gets normed by caller
    all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

    if is_training:
        new_kv_cache = None
    else:
        # Prefill mode: build cache from collected k,v outputs (convert stacked to list)
        keys_list = [all_keys[i] for i in range(num_layers)]
        values_list = [all_values[i] for i in range(num_layers)]
        new_kv_cache = KVCache.from_layer_outputs(keys_list, values_list, attention_mask)

    return final_hs, all_hidden_states, new_kv_cache
