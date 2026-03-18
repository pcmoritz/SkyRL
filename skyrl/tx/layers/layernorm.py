"""Layer normalization implementations."""

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from skyrl.tx.layers.util import Param


def _full_sharding(x: jax.Array) -> NamedSharding:
    sharding = getattr(x, "sharding", None)
    if isinstance(sharding, NamedSharding):
        return sharding
    if x.ndim == 4:
        return NamedSharding(
            jax.sharding.get_abstract_mesh(),
            PartitionSpec("fsdp", None, "tp", None),
        )
    return NamedSharding(
        jax.sharding.get_abstract_mesh(),
        PartitionSpec("fsdp", *([None] * (x.ndim - 1))),
    )


def _reduced_sharding(x: jax.Array) -> NamedSharding:
    sharding = _full_sharding(x)
    spec = tuple(sharding.spec)
    return NamedSharding(sharding.mesh, PartitionSpec(*spec[:-1], None))


@jax.custom_vjp
def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    batch_sharding = _reduced_sharding(x)
    inv_rms = jax.sharding.reshard(
        jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps),
        batch_sharding,
    )
    return weight * x * inv_rms


def _rms_norm_fwd(x: jax.Array, weight: jax.Array, eps: float) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    batch_sharding = _reduced_sharding(x)
    inv_rms = jax.sharding.reshard(
        jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps),
        batch_sharding,
    )
    return weight * x * inv_rms, (x, weight, inv_rms)


def _rms_norm_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array], grad_output: jax.Array
) -> tuple[jax.Array, jax.Array, None]:
    x, weight, inv_rms = residuals
    input_sharding = _full_sharding(x)
    reduced_sharding = _reduced_sharding(x)
    grad_output = jax.sharding.reshard(grad_output, input_sharding)
    weighted_grad = jax.sharding.reshard(grad_output * weight, input_sharding)
    inner = jax.sharding.reshard(jnp.mean(weighted_grad * x, axis=-1, keepdims=True), reduced_sharding)
    grad_x = jax.sharding.reshard(weighted_grad * inv_rms - x * (inv_rms**3) * inner, input_sharding)
    reduce_axes = tuple(range(grad_output.ndim - 1))
    grad_weight = (grad_output * x * inv_rms).sum(axis=reduce_axes)
    return grad_x, grad_weight, None


_rms_norm.defvjp(_rms_norm_fwd, _rms_norm_bwd)


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
        return _rms_norm(x, self.weight, self.eps)
