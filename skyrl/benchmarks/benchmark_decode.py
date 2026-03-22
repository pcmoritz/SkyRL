"""Benchmark decode performance.

Usage:
    # Single batch size:
    python skyrl/benchmarks/benchmark_decode.py

    # Sweep batch sizes:
    python skyrl/benchmarks/benchmark_decode.py --batch-sizes 1,8,32,64,128

    # With JAX profiler trace:
    python skyrl/benchmarks/benchmark_decode.py --profile /tmp/jax-trace
"""

import argparse
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from skyrl.tinker import types
from skyrl.tx.models.configs import Qwen3Config
from skyrl.tx.models.qwen3 import Qwen3ForCausalLM
from skyrl.tx.utils.models import load_safetensors

PROMPT = "Explain the theory of relativity in simple terms"


def bench(model, tokenizer, batch_size, max_tokens, warmup, runs):
    prompts = [PROMPT] * batch_size
    batch = tokenizer(prompts, return_tensors="np", padding=True)
    input_ids = jnp.array(batch.input_ids)
    attention_mask = jnp.array(batch.attention_mask)
    sampling_params = [
        types.SamplingParams(max_tokens=max_tokens, temperature=0.6, seed=42 + i) for i in range(batch_size)
    ]

    for _ in range(warmup):
        result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
        jax.block_until_ready(result.generated_ids)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
        jax.block_until_ready(result.generated_ids)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    total_tokens = batch_size * max_tokens
    return times.mean(), times.std(), total_tokens / times.mean(), times.mean() / max_tokens * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-sizes", default="1,8,32,64,128", help="Comma-separated batch sizes")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--profile", type=str, default=None)
    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.device_count()}")
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '(not set)')}")

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(args.model)
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=True)

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        del hf_model

        mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        param_bytes = sum(p.size * p.dtype.itemsize for p in jax.tree.leaves(nnx.state(model)))
        print(f"Parameters: {param_bytes / 1e6:.0f} MB ({param_bytes / 1e9:.2f} GB)")
        print(f"max_tokens: {args.max_tokens}\n")

        print(f"{'batch':>6} {'ms/step':>8} {'tok/s':>8} {'ms/tok':>8} {'GB/s':>8}")
        print("-" * 46)
        for bs in batch_sizes:
            mean_s, std_s, tok_s, ms_per_step = bench(model, tokenizer, bs, args.max_tokens, args.warmup, args.runs)
            # Approximate bandwidth: weights read once per step
            gbps = param_bytes / (ms_per_step / 1000) / 1e9
            print(f"{bs:>6} {ms_per_step:>8.2f} {tok_s:>8.0f} {mean_s / (bs * args.max_tokens) * 1000:>8.3f} {gbps:>8.1f}")

        if args.profile:
            bs = batch_sizes[-1]
            prompts = [PROMPT] * bs
            batch = tokenizer(prompts, return_tensors="np", padding=True)
            input_ids = jnp.array(batch.input_ids)
            attention_mask = jnp.array(batch.attention_mask)
            sampling_params = [
                types.SamplingParams(max_tokens=args.max_tokens, temperature=0.6, seed=42 + i) for i in range(bs)
            ]
            print(f"\nCollecting profiler trace (batch_size={bs}) to {args.profile}...")
            jax.profiler.start_trace(args.profile)
            result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
            jax.block_until_ready(result.generated_ids)
            jax.profiler.stop_trace()
            print("Trace saved. Open with https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
