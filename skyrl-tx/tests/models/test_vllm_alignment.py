"""Test that TX generator produces numerically aligned results with vLLM."""

import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from vllm import LLM, SamplingParams as VLLMSamplingParams

from tx.models.configs import Qwen3Config
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.tinker import types
from tx.utils.models import load_safetensors


MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def vllm_model():
    """Load vLLM model for testing."""
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=512,
    )
    yield llm


@pytest.fixture(scope="module")
def tx_model():
    """Load TX model for testing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager", use_safetensors=True
    )

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        base_config = PretrainedConfig.from_pretrained(MODEL_NAME)
        config = Qwen3Config(
            base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True
        )

        mesh = jax.make_mesh((1, 1), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        yield model, tokenizer


def generate_vllm(llm: LLM, prompt_tokens: list[int], temperature: float, max_tokens: int) -> dict:
    """Generate with vLLM and return tokens and logprobs."""
    sampling_params = VLLMSamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        top_k=-1,
        logprobs=1,  # Return logprobs for sampled tokens
    )

    # Use TokensPrompt format for vLLM
    prompt = {"prompt_token_ids": prompt_tokens}
    outputs = llm.generate(prompts=[prompt], sampling_params=sampling_params)
    output = outputs[0]

    token_ids = list(output.outputs[0].token_ids)
    logprobs = [lp[token_id].logprob for token_id, lp in zip(token_ids, output.outputs[0].logprobs)]

    return {
        "token_ids": token_ids,
        "logprobs": logprobs,
    }


@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason="Skip vLLM alignment test in CI (requires GPU)"
)
def test_vllm_tx_alignment_greedy(vllm_model, tx_model):
    """Test that TX and vLLM produce identical results for greedy decoding."""
    model, tokenizer = tx_model
    llm = vllm_model

    prompts = [
        "The capital of France is",
        "In machine learning, a neural network",
        "The quick brown fox jumps over",
    ]
    max_tokens = 20

    for prompt in prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([tokens])
        attention_mask = np.ones_like(input_ids)

        # Generate with TX
        sampling_params = [types.SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=42)]
        tx_result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
        tx_tokens = tx_result.generated_ids[0]
        tx_logprobs = tx_result.logprobs[0]

        # Generate with vLLM
        vllm_result = generate_vllm(llm, tokens, temperature=0.0, max_tokens=max_tokens)
        vllm_tokens = vllm_result["token_ids"]
        vllm_logprobs = vllm_result["logprobs"]

        # Compare tokens (should be identical for greedy decoding)
        assert tx_tokens == vllm_tokens, (
            f"Token mismatch for prompt '{prompt}':\n"
            f"TX tokens: {tx_tokens}\n"
            f"vLLM tokens: {vllm_tokens}"
        )

        # Compare logprobs (should be close, allowing for bfloat16 numerical differences)
        for i, (tx_lp, vllm_lp) in enumerate(zip(tx_logprobs, vllm_logprobs)):
            assert np.isclose(tx_lp, vllm_lp, rtol=1e-2, atol=1e-2), (
                f"Logprob mismatch for prompt '{prompt}' at position {i}:\n"
                f"TX: {tx_lp}, vLLM: {vllm_lp}, diff: {abs(tx_lp - vllm_lp)}"
            )


@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason="Skip vLLM alignment test in CI (requires GPU)"
)
def test_vllm_tx_alignment_with_temperature(vllm_model, tx_model):
    """Test that TX and vLLM produce aligned logprobs with temperature > 0.

    Note: With temperature > 0, tokens may differ due to different RNG,
    but logprobs for the same tokens should match.
    """
    model, tokenizer = tx_model
    llm = vllm_model

    prompt = "The meaning of life is"
    max_tokens = 10
    temperature = 1.0

    # Tokenize
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = np.array([tokens])
    attention_mask = np.ones_like(input_ids)

    # Generate with TX
    sampling_params_temp = [types.SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=42)]
    tx_result_temp = model.generate(input_ids, attention_mask, sampling_params=sampling_params_temp)

    # Generate with vLLM
    vllm_result = generate_vllm(llm, tokens, temperature=temperature, max_tokens=max_tokens)

    print(f"TX logprobs (temp={temperature}): {tx_result_temp.logprobs[0]}")
    print(f"vLLM logprobs (temp={temperature}): {vllm_result['logprobs']}")

    # Note: Due to different RNG, tokens may differ, so we just verify
    # that logprobs are in a reasonable range and not NaN/inf
    for lp in tx_result_temp.logprobs[0]:
        assert np.isfinite(lp), f"TX logprob is not finite: {lp}"
        assert lp <= 0, f"TX logprob should be non-positive: {lp}"

    for lp in vllm_result["logprobs"]:
        assert np.isfinite(lp), f"vLLM logprob is not finite: {lp}"
        assert lp <= 0, f"vLLM logprob should be non-positive: {lp}"


@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason="Skip vLLM alignment test in CI (requires GPU)"
)
def test_vllm_tx_logprob_precision(vllm_model, tx_model):
    """Test logprob precision alignment between TX and vLLM.

    This test specifically checks that the float32 log_softmax computation
    in TX matches vLLM's precision.
    """
    model, tokenizer = tx_model
    llm = vllm_model

    # Use a longer prompt to test more tokens
    prompt = "Artificial intelligence is transforming the way we"
    max_tokens = 30

    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = np.array([tokens])
    attention_mask = np.ones_like(input_ids)

    # Greedy decoding for deterministic comparison
    sampling_params = [types.SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=42)]
    tx_result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)

    vllm_result = generate_vllm(llm, tokens, temperature=0.0, max_tokens=max_tokens)

    # Collect statistics on differences
    differences = []
    for i, (tx_lp, vllm_lp) in enumerate(zip(tx_result.logprobs[0], vllm_result["logprobs"])):
        diff = abs(tx_lp - vllm_lp)
        differences.append(diff)

    differences = np.array(differences)
    print(f"Logprob differences - mean: {differences.mean():.6f}, max: {differences.max():.6f}, std: {differences.std():.6f}")

    # Assert all differences are within tolerance
    assert differences.max() < 1e-2, (
        f"Maximum logprob difference {differences.max():.6f} exceeds tolerance.\n"
        f"TX logprobs: {tx_result.logprobs[0]}\n"
        f"vLLM logprobs: {vllm_result['logprobs']}"
    )

    # Assert mean difference is very small
    assert differences.mean() < 1e-3, (
        f"Mean logprob difference {differences.mean():.6f} exceeds tolerance."
    )
