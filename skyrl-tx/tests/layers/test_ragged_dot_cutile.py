"""Tests for cuTile ragged_dot kernel with group_offset support."""

import jax
import jax.numpy as jnp
import pytest

# Skip all tests if cuda-tile or cudart is not available
pytest.importorskip("cuda.tile")
pytest.importorskip("cuda.cudart")

from tx.layers.ragged_dot_cutile import ragged_dot, gmm, tgmm


@pytest.mark.parametrize(
    "group_sizes,group_offset,g_local,expected_scale",
    [
        ([2, 2, 2], 1, 2, [0, 0, 1, 1, 2, 2]),  # middle shard
        ([2, 2, 2], 0, 2, [1, 1, 2, 2, 0, 0]),  # first shard
        ([2, 2, 2], 2, 1, [0, 0, 0, 0, 1, 1]),  # last shard
        ([6], 0, 1, [1, 1, 1, 1, 1, 1]),  # single group
        ([2, 0, 0, 4], 1, 2, [0, 0, 0, 0, 0, 0]),  # empty groups in shard
        ([1, 3, 2], 1, 2, [0, 1, 1, 1, 2, 2]),  # uneven sizes
    ],
)
def test_ragged_dot_cutile_with_group_offset(group_sizes, group_offset, g_local, expected_scale):
    """Test cuTile ragged_dot with group_offset for various edge cases."""
    group_sizes = jnp.array(group_sizes)
    m, d = 6, 2

    lhs = jnp.arange(m * d, dtype=jnp.float32).reshape(m, d)
    rhs = jnp.stack([(i + 1) * jnp.eye(d) for i in range(g_local)])  # 1*I, 2*I, ...

    result = gmm(lhs, rhs, group_sizes, group_offset=jnp.array([group_offset]))

    # expected_scale: 0 for masked tokens, else local_group_idx + 1
    scale = jnp.array(expected_scale, dtype=jnp.float32)[:, None]
    expected = lhs * scale

    assert jnp.allclose(result, expected), f"Got:\n{result}\nExpected:\n{expected}"


def test_ragged_dot_cutile_basic():
    """Test basic grouped matmul without group_offset."""
    m, k, n = 128, 64, 32
    num_groups = 4
    group_sizes = jnp.array([32, 32, 32, 32], dtype=jnp.int32)

    lhs = jax.random.normal(jax.random.key(0), (m, k))
    rhs = jax.random.normal(jax.random.key(1), (num_groups, k, n))

    result = gmm(lhs, rhs, group_sizes)

    # Compute reference using manual slicing
    expected = jnp.zeros((m, n))
    offset = 0
    for g in range(num_groups):
        size = group_sizes[g]
        expected = expected.at[offset : offset + size].set(lhs[offset : offset + size] @ rhs[g])
        offset += size

    assert jnp.allclose(result, expected, atol=1e-5), f"Max diff: {jnp.abs(result - expected).max()}"


def test_tgmm_basic():
    """Test transposed grouped matmul for backward pass."""
    m, k, n = 64, 32, 16
    num_groups = 2
    group_sizes = jnp.array([32, 32], dtype=jnp.int32)

    lhs = jax.random.normal(jax.random.key(0), (k, m))  # [k, m]
    rhs = jax.random.normal(jax.random.key(1), (m, n))  # [m, n]

    result = tgmm(lhs, rhs, group_sizes)

    # Reference: out[g] = lhs[:, group_start:group_end] @ rhs[group_start:group_end, :]
    expected = jnp.zeros((num_groups, k, n))
    offset = 0
    for g in range(num_groups):
        size = group_sizes[g]
        expected = expected.at[g].set(lhs[:, offset : offset + size] @ rhs[offset : offset + size])
        offset += size

    assert jnp.allclose(result, expected, atol=1e-5), f"Max diff: {jnp.abs(result - expected).max()}"


def test_ragged_dot_cutile_vjp():
    """Test automatic differentiation through ragged_dot."""
    m, k, n = 64, 32, 16
    num_groups = 2
    group_sizes = jnp.array([32, 32], dtype=jnp.int32)

    lhs = jax.random.normal(jax.random.key(0), (m, k))
    rhs = jax.random.normal(jax.random.key(1), (num_groups, k, n))

    def loss_fn(lhs, rhs):
        out = ragged_dot(lhs, rhs, group_sizes)
        return jnp.sum(out**2)

    # Test that gradients can be computed
    grad_lhs, grad_rhs = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)

    assert grad_lhs.shape == lhs.shape
    assert grad_rhs.shape == rhs.shape
    assert not jnp.any(jnp.isnan(grad_lhs))
    assert not jnp.any(jnp.isnan(grad_rhs))


def test_ragged_dot_cutile_with_group_offset_vjp():
    """Test autodiff with group_offset."""
    m, k, n = 64, 32, 16
    group_sizes = jnp.array([16, 16, 16, 16], dtype=jnp.int32)
    num_local = 2
    group_offset = jnp.array([1])  # Process groups 1 and 2

    lhs = jax.random.normal(jax.random.key(0), (m, k))
    rhs = jax.random.normal(jax.random.key(1), (num_local, k, n))

    def loss_fn(lhs, rhs):
        out = ragged_dot(lhs, rhs, group_sizes, group_offset=group_offset)
        return jnp.sum(out**2)

    grad_lhs, grad_rhs = jax.grad(loss_fn, argnums=(0, 1))(lhs, rhs)

    assert grad_lhs.shape == lhs.shape
    assert grad_rhs.shape == rhs.shape
