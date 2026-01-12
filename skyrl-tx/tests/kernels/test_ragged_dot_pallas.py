"""Tests for Pallas-based ragged_dot kernel with group_offset support."""

import jax
import jax.numpy as jnp
import pytest

from tx.kernels.ragged_dot import ragged_dot_pallas, _ragged_dot_simple, _trans_ragged_dot_simple
from tx.layers.util import _ragged_dot_fallback


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
def test_ragged_dot_pallas_forward(group_sizes, group_offset, g_local, expected_scale):
    """Test Pallas kernel forward pass matches expected output."""
    group_sizes = jnp.array(group_sizes)
    m, d = 6, 2

    lhs = jnp.arange(m * d, dtype=jnp.float32).reshape(m, d)
    rhs = jnp.stack([(i + 1) * jnp.eye(d) for i in range(g_local)])  # 1*I, 2*I, ...

    result = jax.jit(ragged_dot_pallas)(lhs, rhs, group_sizes, jnp.array([group_offset]))

    # expected_scale: 0 for masked tokens, else local_group_idx + 1
    scale = jnp.array(expected_scale, dtype=jnp.float32)[:, None]
    expected = lhs * scale

    assert jnp.allclose(result, expected), f"Got:\n{result}\nExpected:\n{expected}"


@pytest.mark.parametrize(
    "group_sizes,group_offset,g_local",
    [
        ([2, 2, 2], 1, 2),  # middle shard
        ([2, 2, 2], 0, 2),  # first shard
        ([2, 2, 2], 2, 1),  # last shard
        ([6], 0, 1),  # single group
        ([1, 3, 2], 1, 2),  # uneven sizes
    ],
)
def test_ragged_dot_pallas_matches_fallback(group_sizes, group_offset, g_local):
    """Test Pallas kernel matches fallback implementation."""
    group_sizes = jnp.array(group_sizes)
    m, k, n = 6, 4, 3

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(key2, (g_local, k, n), dtype=jnp.float32)
    group_offset_arr = jnp.array([group_offset])

    result_pallas = ragged_dot_pallas(lhs, rhs, group_sizes, group_offset_arr)
    result_fallback = _ragged_dot_fallback(lhs, rhs, group_sizes, group_offset_arr)

    assert jnp.allclose(result_pallas, result_fallback, atol=1e-5), (
        f"Pallas:\n{result_pallas}\nFallback:\n{result_fallback}"
    )


@pytest.mark.parametrize(
    "group_sizes,group_offset,g_local",
    [
        ([2, 2, 2], 1, 2),  # middle shard
        ([2, 2, 2], 0, 2),  # first shard
        ([2, 2, 2], 2, 1),  # last shard
        ([6], 0, 1),  # single group
    ],
)
def test_ragged_dot_pallas_gradient_lhs(group_sizes, group_offset, g_local):
    """Test gradient w.r.t. lhs is correct."""
    group_sizes = jnp.array(group_sizes)
    m, k, n = 6, 4, 3

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(key2, (g_local, k, n), dtype=jnp.float32)
    group_offset_arr = jnp.array([group_offset])

    # Compute gradient via autodiff
    def loss_fn(lhs):
        out = ragged_dot_pallas(lhs, rhs, group_sizes, group_offset_arr)
        return jnp.sum(out**2)

    grad_autodiff = jax.grad(loss_fn)(lhs)

    # Compute gradient via finite differences
    eps = 1e-4
    grad_fd = jnp.zeros_like(lhs)
    for i in range(m):
        for j in range(k):
            lhs_plus = lhs.at[i, j].add(eps)
            lhs_minus = lhs.at[i, j].add(-eps)
            loss_plus = loss_fn(lhs_plus)
            loss_minus = loss_fn(lhs_minus)
            grad_fd = grad_fd.at[i, j].set((loss_plus - loss_minus) / (2 * eps))

    # Use relative tolerance check since absolute differences depend on gradient magnitude
    assert jnp.allclose(grad_autodiff, grad_fd, rtol=1e-2, atol=1e-2), (
        f"Autodiff grad:\n{grad_autodiff}\nFinite diff:\n{grad_fd}"
    )


@pytest.mark.parametrize(
    "group_sizes,group_offset,g_local",
    [
        ([2, 2, 2], 1, 2),  # middle shard
        ([2, 2, 2], 0, 2),  # first shard
        ([6], 0, 1),  # single group
    ],
)
def test_ragged_dot_pallas_gradient_rhs(group_sizes, group_offset, g_local):
    """Test gradient w.r.t. rhs is correct."""
    group_sizes = jnp.array(group_sizes)
    m, k, n = 6, 4, 3

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(key2, (g_local, k, n), dtype=jnp.float32)
    group_offset_arr = jnp.array([group_offset])

    # Compute gradient via autodiff
    def loss_fn(rhs):
        out = ragged_dot_pallas(lhs, rhs, group_sizes, group_offset_arr)
        return jnp.sum(out**2)

    grad_autodiff = jax.grad(loss_fn)(rhs)

    # Compute gradient via finite differences
    eps = 1e-4
    grad_fd = jnp.zeros_like(rhs)
    for i in range(g_local):
        for j in range(k):
            for l in range(n):
                rhs_plus = rhs.at[i, j, l].add(eps)
                rhs_minus = rhs.at[i, j, l].add(-eps)
                loss_plus = loss_fn(rhs_plus)
                loss_minus = loss_fn(rhs_minus)
                grad_fd = grad_fd.at[i, j, l].set((loss_plus - loss_minus) / (2 * eps))

    # Use relative tolerance check since absolute differences depend on gradient magnitude
    assert jnp.allclose(grad_autodiff, grad_fd, rtol=1e-2, atol=1e-2), (
        f"Autodiff grad:\n{grad_autodiff}\nFinite diff:\n{grad_fd}"
    )


def test_trans_ragged_dot_simple():
    """Test transpose ragged dot (used in backward pass)."""
    m, k, n = 6, 4, 3
    g = 3
    g_local = 2
    group_offset = 1
    group_sizes = jnp.array([2, 2, 2])

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(key2, (m, n), dtype=jnp.float32)
    group_offset_arr = jnp.array([group_offset])

    result = _trans_ragged_dot_simple(lhs, rhs, group_sizes, group_offset_arr, g_local)

    # Verify shape
    assert result.shape == (g_local, k, n), f"Expected shape {(g_local, k, n)}, got {result.shape}"

    # Manual computation for verification
    # Tokens 2,3 belong to group 1 (local index 0)
    # Tokens 4,5 belong to group 2 (local index 1)
    expected = jnp.zeros((g_local, k, n))
    # Group 1 (local 0): tokens 2, 3
    expected = expected.at[0].set(lhs[2:4].T @ rhs[2:4])
    # Group 2 (local 1): tokens 4, 5
    expected = expected.at[1].set(lhs[4:6].T @ rhs[4:6])

    assert jnp.allclose(result, expected, atol=1e-5), f"Got:\n{result}\nExpected:\n{expected}"


def test_ragged_dot_pallas_jit():
    """Test that Pallas kernel works under JIT."""
    group_sizes = jnp.array([2, 2, 2])
    m, k, n = 6, 4, 3
    g_local = 2
    group_offset = 1

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(key2, (g_local, k, n), dtype=jnp.float32)
    group_offset_arr = jnp.array([group_offset])

    # Non-JIT result
    result_eager = ragged_dot_pallas(lhs, rhs, group_sizes, group_offset_arr)

    # JIT result
    result_jit = jax.jit(ragged_dot_pallas)(lhs, rhs, group_sizes, group_offset_arr)

    assert jnp.allclose(result_eager, result_jit, atol=1e-5)


def test_ragged_dot_pallas_bfloat16():
    """Test Pallas kernel with bfloat16 inputs."""
    group_sizes = jnp.array([2, 2, 2])
    m, k, n = 6, 4, 3
    g_local = 2
    group_offset = 1

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key2, (g_local, k, n), dtype=jnp.bfloat16)
    group_offset_arr = jnp.array([group_offset])

    result = ragged_dot_pallas(lhs, rhs, group_sizes, group_offset_arr)

    assert result.dtype == jnp.bfloat16
    assert result.shape == (m, n)


def test_ragged_dot_pallas_larger_problem():
    """Test with larger problem size."""
    g = 8
    m = 64
    k = 32
    n = 16
    g_local = 4
    group_offset = 2

    # Create group sizes that sum to m
    tokens_per_group = m // g
    group_sizes = jnp.array([tokens_per_group] * g)

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    lhs = jax.random.normal(key1, (m, k), dtype=jnp.float32)
    rhs = jax.random.normal(key2, (g_local, k, n), dtype=jnp.float32)
    group_offset_arr = jnp.array([group_offset])

    result = ragged_dot_pallas(lhs, rhs, group_sizes, group_offset_arr)

    # Verify basic properties
    assert result.shape == (m, n)

    # Verify only tokens in groups [2, 3, 4, 5] are non-zero
    start_token = group_offset * tokens_per_group
    end_token = (group_offset + g_local) * tokens_per_group

    # Tokens outside range should be zero
    assert jnp.allclose(result[:start_token], 0.0)
    assert jnp.allclose(result[end_token:], 0.0)

    # Tokens inside range should be non-zero (with high probability)
    assert jnp.any(result[start_token:end_token] != 0.0)
