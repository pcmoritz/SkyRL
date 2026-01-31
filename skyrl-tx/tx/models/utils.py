"""Utility functions for model forward passes with stacked decoder layers.

This module provides:
- create_stacked_layers: Create decoder layers with stacked weights using nnx.vmap
- forward_layers: Unified forward pass using scan (skips KV cache during training)

Prerequisites:
- Layers must be created with nnx.vmap (stacked weights)
- KVCache must use stacked format: (num_layers, batch, seq, heads, dim)
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

    Uses scan for prefill/training (efficient for long sequences) and unrolled
    Python loop for decode (avoids nested scan issues with outer decode loop).

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
        # Decode mode: use unrolled Python loop to avoid nested scan issues.
        # When this runs inside the outer decode loop (token generation), using
        # scan here would cause XLA to pass all layer weights as loop-carried
        # state (iterArgs) in the outer loop, hurting performance.
        return _forward_layers_decode(
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
    else:
        # Prefill/training mode: use scan for efficiency with long sequences.
        return _forward_layers_scan(
            layer_graphdef,
            layer_state,
            hidden_states,
            num_layers,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            output_hidden_states=output_hidden_states,
            gradient_checkpointing=gradient_checkpointing,
            is_training=is_training,
        )


def _forward_layers_decode(
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
    """Decode forward pass using unrolled Python loop.

    This avoids nested scan issues when called from within the outer decode loop.
    The Python loop is unrolled at trace time, making layer weights appear as
    constants rather than loop-carried state to XLA.
    """
    all_hidden_states = []
    updated_keys = []
    updated_values = []

    for layer_idx in range(num_layers):
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Extract this layer's parameters
        layer_params = jax.tree.map(lambda x: x[layer_idx], layer_state)
        layer = nnx.merge(layer_graphdef, layer_params)

        # Get this layer's KV cache
        layer_kv = (kv_cache.keys[layer_idx], kv_cache.values[layer_idx])

        hidden_states, (k, v) = layer(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )
        updated_keys.append(k)
        updated_values.append(v)

    new_kv_cache = KVCache(
        keys=jax.numpy.stack(updated_keys),
        values=jax.numpy.stack(updated_values),
        cache_position=kv_cache.cache_position + positions.shape[1],
    )

    return hidden_states, all_hidden_states, new_kv_cache


def _forward_layers_scan(
    layer_graphdef: nnx.GraphDef,
    layer_state: nnx.State,
    hidden_states: jax.Array,
    num_layers: int,
    *,
    attention_mask: jax.Array,
    positions: jax.Array,
    adapter_indices: jax.Array | None,
    output_hidden_states: bool,
    gradient_checkpointing: bool,
    is_training: bool,
) -> tuple[jax.Array, list[jax.Array], KVCache | None]:
    """Prefill/training forward pass using scan.

    Efficient for long sequences. Pass layer_state through xs so scan handles
    slicing automatically, avoiding closure capture issues with gradient checkpointing.
    """

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
            # Avoid accumulating large KV tensors for training.
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
        # Prefill mode: build cache from collected k,v outputs
        new_kv_cache = KVCache.from_layer_outputs(all_keys, all_values, attention_mask)

    return final_hs, all_hidden_states, new_kv_cache
