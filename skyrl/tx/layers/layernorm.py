"""Layer normalization implementations."""

import jax
from flax import nnx
from jax import numpy as jnp

from skyrl.tx.layers.util import Param


class RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, size: int, *, eps: float = 1e-6, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(
            size, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P(None)), rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.weight * x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
