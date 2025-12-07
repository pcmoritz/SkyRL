"""Test to find where TX and vLLM diverge for Qwen3-4B."""

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


MODEL_NAME = "Qwen/Qwen3-4B"


@pytest.fixture(scope="module")
def vllm_model():
    """Load vLLM model for testing."""
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=1024,
        tensor_parallel_size=4,
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

        mesh = jax.make_mesh((1, 4), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        yield model, tokenizer


@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason="Skip in CI (requires GPU and large model)"
)
def test_qwen3_4b_first_token_alignment(vllm_model, tx_model):
    """Test if first token matches between TX and vLLM for Qwen3-4B."""
    model, tokenizer = tx_model
    llm = vllm_model

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In machine learning, gradient descent",
        "The quick brown fox",
        "Write a Python function to sort a list:",
    ]

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([tokens])
        attention_mask = np.ones_like(input_ids)

        # TX generation
        sampling_params = [types.SamplingParams(max_tokens=20, temperature=0.0, seed=42)]
        tx_result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
        tx_tokens = tx_result.generated_ids[0]

        # vLLM generation
        vllm_params = VLLMSamplingParams(temperature=0.0, max_tokens=20)
        vllm_output = llm.generate([{"prompt_token_ids": tokens}], sampling_params=vllm_params)
        vllm_tokens = list(vllm_output[0].outputs[0].token_ids)

        # Find divergence point
        diverge_at = None
        for i in range(min(len(tx_tokens), len(vllm_tokens))):
            if tx_tokens[i] != vllm_tokens[i]:
                diverge_at = i
                break

        print(f"\nPrompt: {prompt!r}")
        print(f"TX tokens:   {tx_tokens}")
        print(f"vLLM tokens: {vllm_tokens}")
        print(f"TX text:   {tokenizer.decode(tx_tokens)!r}")
        print(f"vLLM text: {tokenizer.decode(vllm_tokens)!r}")

        if diverge_at is not None:
            print(f"DIVERGE at token {diverge_at}: TX={tx_tokens[diverge_at]}, vLLM={vllm_tokens[diverge_at]}")
            print(f"  TX token:   {tokenizer.decode([tx_tokens[diverge_at]])!r}")
            print(f"  vLLM token: {tokenizer.decode([vllm_tokens[diverge_at]])!r}")
        else:
            print("MATCH: All tokens identical")

        # Assert first token matches at minimum
        assert tx_tokens[0] == vllm_tokens[0], (
            f"First token mismatch for {prompt!r}: TX={tx_tokens[0]}, vLLM={vllm_tokens[0]}"
        )


@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason="Skip in CI (requires GPU and large model)"
)
def test_qwen3_4b_logits_comparison(vllm_model, tx_model):
    """Compare raw logits between TX and vLLM for first token."""
    model, tokenizer = tx_model
    llm = vllm_model

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = np.array([tokens])
    attention_mask = np.ones_like(input_ids)

    # TX: get logits from prefill
    positions = jnp.arange(len(tokens))[None, :]
    tx_output = model(input_ids, attention_mask=attention_mask, positions=positions)
    tx_logits = np.array(tx_output.logits[0, -1, :])  # Last position logits

    # Get top-10 from TX
    tx_top10_idx = np.argsort(tx_logits)[-10:][::-1]
    tx_top10_logits = tx_logits[tx_top10_idx]

    # vLLM: get logprobs for top tokens
    vllm_params = VLLMSamplingParams(temperature=0.0, max_tokens=1, logprobs=10)
    vllm_output = llm.generate([{"prompt_token_ids": tokens}], sampling_params=vllm_params)

    vllm_first_token = vllm_output[0].outputs[0].token_ids[0]
    vllm_logprobs = vllm_output[0].outputs[0].logprobs[0]

    print(f"\nPrompt: {prompt!r}")
    print(f"\nTX top-10 tokens:")
    for idx, logit in zip(tx_top10_idx, tx_top10_logits):
        print(f"  {idx:6d} ({tokenizer.decode([idx]):>15s}): {logit:.4f}")

    print(f"\nvLLM first token: {vllm_first_token} ({tokenizer.decode([vllm_first_token])!r})")
    print(f"vLLM top logprobs:")
    for token_id, lp_info in vllm_logprobs.items():
        print(f"  {token_id:6d} ({tokenizer.decode([token_id]):>15s}): {lp_info.logprob:.4f}")

    tx_argmax = tx_top10_idx[0]
    print(f"\nTX argmax: {tx_argmax} ({tokenizer.decode([tx_argmax])!r})")
    print(f"vLLM argmax: {vllm_first_token} ({tokenizer.decode([vllm_first_token])!r})")
    print(f"Match: {tx_argmax == vllm_first_token}")
