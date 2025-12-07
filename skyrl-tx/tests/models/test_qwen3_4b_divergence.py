"""Test to find where TX and vLLM diverge for Qwen3-4B."""

import gc
import os
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from vllm import LLM, SamplingParams as VLLMSamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from tx.models.configs import Qwen3Config
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.tinker import types
from tx.utils.models import load_safetensors


MODEL_NAME = "Qwen/Qwen3-4B"


def run_vllm_generation(prompts: list[str], tokenizer, max_tokens: int = 20) -> dict:
    """Run vLLM generation and return results, then clean up."""
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=1024,
        tensor_parallel_size=4,
    )

    results = {}
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        vllm_params = VLLMSamplingParams(temperature=0.0, max_tokens=max_tokens)
        vllm_output = llm.generate([{"prompt_token_ids": tokens}], sampling_params=vllm_params)
        vllm_tokens = list(vllm_output[0].outputs[0].token_ids)
        results[prompt] = {"tokens": tokens, "generated": vllm_tokens}

    # Clean up vLLM
    destroy_model_parallel()
    del llm
    gc.collect()

    # Try to clear CUDA cache if torch is available
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return results


def run_tx_generation(prompts: list[str], vllm_results: dict, tokenizer, max_tokens: int = 20):
    """Run TX generation and compare with vLLM results."""
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager", use_safetensors=True
    )

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        del hf_model
        gc.collect()

        base_config = PretrainedConfig.from_pretrained(MODEL_NAME)
        config = Qwen3Config(
            base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True
        )

        mesh = jax.make_mesh((1, 4), ("dp", "tp"))
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        for prompt in prompts:
            tokens = vllm_results[prompt]["tokens"]
            vllm_tokens = vllm_results[prompt]["generated"]

            input_ids = np.array([tokens])
            attention_mask = np.ones_like(input_ids)

            # TX generation
            sampling_params = [types.SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=42)]
            tx_result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
            tx_tokens = tx_result.generated_ids[0]

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


@pytest.mark.skipif(
    os.environ.get("CI") is not None,
    reason="Skip in CI (requires GPU and large model)"
)
def test_qwen3_4b_first_token_alignment():
    """Test if first token matches between TX and vLLM for Qwen3-4B."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In machine learning, gradient descent",
        "The quick brown fox",
        "Write a Python function to sort a list:",
    ]

    # Generate 256 tokens to match the actual use case
    max_tokens = 256

    # First run vLLM and collect results
    print("\n=== Running vLLM ===")
    vllm_results = run_vllm_generation(prompts, tokenizer, max_tokens=max_tokens)

    # Then run TX and compare
    print("\n=== Running TX ===")
    run_tx_generation(prompts, vllm_results, tokenizer, max_tokens=max_tokens)
