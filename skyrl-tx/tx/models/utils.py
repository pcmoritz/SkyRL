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
import jax.numpy as jnp

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

    # For decode mode without gradient checkpointing, use a Python loop instead
    # of scan. This allows XLA to optimize memory reuse between layer calls,
    # avoiding the memory overhead of scan's output collection.
    if is_decode and not gradient_checkpointing:
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

    def body_fn(hs, xs):
        # Unpack xs: scan automatically slices the leading dimension of layer_state
        if is_decode:
            layer_params, layer_k, layer_v = xs
            layer_kv = (layer_k, layer_v)
        else:
            layer_params = xs
            layer_kv = None

        # Merge using the sliced params directly - no manual gather needed
        layer = nnx.merge(layer_graphdef, layer_params)
        new_hs, (k, v) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )
        hs_output = new_hs if output_hidden_states else None

        if is_training:
            # Avoid accumulating large KV tensors for training.
            k = v = None
        # Note: During decode, layers now return only the new K/V values
        # (batch, 1, heads, dim) instead of the full cache, so no extraction needed.

        return new_hs, (hs_output, k, v)

    if gradient_checkpointing:
        body_fn = jax.checkpoint(body_fn)

    # Pass layer_state as xs so scan handles the slicing automatically.
    # This avoids capturing layer_state as a closure and manually gathering,
    # which causes slow XLA compilation with jax.checkpoint.
    xs = (layer_state, kv_cache.keys, kv_cache.values) if is_decode else layer_state

    final_hs, (all_hs, all_keys, all_values) = jax.lax.scan(body_fn, hidden_states, xs)

    # [embed, layer0_out, ..., layer(N-2)_out]; final layer output gets normed by caller
    all_hidden_states = [hidden_states] + list(all_hs[:-1]) if output_hidden_states else []

    if is_training:
        new_kv_cache = None
    elif is_decode:
        # Decode mode: all_keys/all_values contain only the new K/V slices
        # with shape (num_layers, batch, 1, heads, dim). Update the original
        # cache using scatter-style .at[].set() which avoids copies from moveaxis.
        def update_cache_scatter(cache, new_vals, cache_positions):
            # cache: (num_layers, batch, seq, heads, dim)
            # new_vals: (num_layers, batch, 1, heads, dim)
            # cache_positions: (batch,)
            num_layers, batch_size = cache.shape[0], cache.shape[1]

            # Create index arrays for scatter update
            layer_idx = jnp.arange(num_layers)[:, None]  # (num_layers, 1)
            batch_idx = jnp.arange(batch_size)[None, :]  # (1, batch)

            # Broadcast to (num_layers, batch)
            layer_idx = jnp.broadcast_to(layer_idx, (num_layers, batch_size))
            batch_idx = jnp.broadcast_to(batch_idx, (num_layers, batch_size))
            pos_idx = jnp.broadcast_to(cache_positions[None, :], (num_layers, batch_size))

            # new_vals is (num_layers, batch, 1, heads, dim), squeeze the seq dim
            new_vals_squeezed = new_vals[:, :, 0, :, :]  # (num_layers, batch, heads, dim)

            # Scatter update - much more efficient than moveaxis + vmap
            return cache.at[layer_idx, batch_idx, pos_idx].set(new_vals_squeezed)

        new_kv_cache = KVCache(
            keys=update_cache_scatter(kv_cache.keys, all_keys, kv_cache.cache_position),
            values=update_cache_scatter(kv_cache.values, all_values, kv_cache.cache_position),
            cache_position=kv_cache.cache_position + positions.shape[1],
        )
    else:
        # Prefill mode: build cache from collected k,v outputs
        new_kv_cache = KVCache.from_layer_outputs(all_keys, all_values, attention_mask)

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
    """Decode using fori_loop with cache accessed via closure.

    Key optimization: The large KV cache is NOT passed through the loop carry.
    Instead, we only pass small accumulators for the new k/v values.
    The cache is read from closure (read-only during the loop).
    """
    cache_pos = kv_cache.cache_position
    batch_size = hidden_states.shape[0]
    batch_idx = jnp.arange(batch_size)

    # Get cache shape info for initializing accumulators
    # kv_cache.keys shape: (num_layers, batch, seq, heads, dim)
    _, _, _, num_heads, head_dim = kv_cache.keys.shape

    # Initialize small accumulators for new k/v values only
    # Shape: (num_layers, batch, heads, dim) - no seq dimension
    new_keys_accum = jnp.zeros((num_layers, batch_size, num_heads, head_dim), dtype=kv_cache.keys.dtype)
    new_values_accum = jnp.zeros((num_layers, batch_size, num_heads, head_dim), dtype=kv_cache.values.dtype)

    def body_fn(i, carry):
        hs, new_k_acc, new_v_acc = carry

        # Get layer i's parameters by dynamic indexing
        layer_params_i = jax.tree.map(lambda x: x[i], layer_state)
        layer = nnx.merge(layer_graphdef, layer_params_i)

        # Read layer i's cache from closure (NOT in carry)
        layer_kv = (kv_cache.keys[i], kv_cache.values[i])

        # Forward through layer
        new_hs, (k_new, v_new) = layer(
            hs,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=layer_kv,
        )

        # Accumulate new k/v values (small tensors only)
        # k_new, v_new are (batch, 1, heads, dim)
        new_k_acc = new_k_acc.at[i].set(k_new[:, 0, :, :])
        new_v_acc = new_v_acc.at[i].set(v_new[:, 0, :, :])

        return (new_hs, new_k_acc, new_v_acc)

    init_carry = (hidden_states, new_keys_accum, new_values_accum)
    final_hs, all_new_keys, all_new_values = jax.lax.fori_loop(
        0, num_layers, body_fn, init_carry
    )

    # Update the original cache with all new k/v values in one batched operation
    # all_new_keys shape: (num_layers, batch, heads, dim)
    layer_idx = jnp.arange(num_layers)[:, None]
    batch_idx_2d = jnp.arange(batch_size)[None, :]
    layer_idx = jnp.broadcast_to(layer_idx, (num_layers, batch_size))
    batch_idx_2d = jnp.broadcast_to(batch_idx_2d, (num_layers, batch_size))
    pos_idx = jnp.broadcast_to(cache_pos[None, :], (num_layers, batch_size))

    new_kv_cache = KVCache(
        keys=kv_cache.keys.at[layer_idx, batch_idx_2d, pos_idx].set(all_new_keys),
        values=kv_cache.values.at[layer_idx, batch_idx_2d, pos_idx].set(all_new_values),
        cache_position=cache_pos + positions.shape[1],
    )

    # output_hidden_states not supported with fori_loop (would require scan)
    all_hidden_states = []

    return final_hs, all_hidden_states, new_kv_cache
