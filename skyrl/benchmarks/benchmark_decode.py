"""Benchmark decode performance and verify XLA buffer donation / command buffers.

Usage:
    # Baseline:
    python -m skyrl.benchmarks.benchmark_decode

    # With XLA command buffers (reduces kernel launch overhead):
    XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUDNN,CUBLAS" \
        python -m skyrl.benchmarks.benchmark_decode

    # With JAX profiler trace (inspect in chrome://tracing or Perfetto):
    python -m skyrl.benchmarks.benchmark_decode --profile /tmp/jax-trace

    # Dump XLA HLO to inspect buffer donation:
    XLA_FLAGS="--xla_dump_to=/tmp/xla-dump --xla_dump_hlo_as_text" \
        python -m skyrl.benchmarks.benchmark_decode
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 decode performance")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=5, help="Timed iterations")
    parser.add_argument("--profile", type=str, default=None, help="Path to save JAX profiler trace")
    args = parser.parse_args()

    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.device_count()}")
    print(f"XLA_FLAGS: {jax._src.config.FLAGS.jax_xla_backend or 'default'}")
    print()

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=True)
    base_config = PretrainedConfig.from_pretrained(args.model)
    config = Qwen3Config(base_config, max_lora_adapters=0, max_lora_rank=0, shard_attention_heads=True)

    prompts = [
        "Explain the theory of relativity in simple terms",
        "Write a short story about a robot learning to paint",
        "What are the main differences between Python and Rust",
        "Describe how neural networks learn from data",
        "What causes the northern lights to appear in the sky",
        "How do computers store and retrieve information",
        "Explain the water cycle and its importance to life",
        "What is the history of the internet and web",
    ]
    prompts = prompts[: args.batch_size]
    if len(prompts) < args.batch_size:
        prompts = prompts * (args.batch_size // len(prompts) + 1)
        prompts = prompts[: args.batch_size]

    batch = tokenizer(prompts, return_tensors="np", padding=True)
    input_ids = jnp.array(batch.input_ids)
    attention_mask = jnp.array(batch.attention_mask)
    prompt_len = input_ids.shape[1]

    sampling_params = [
        types.SamplingParams(max_tokens=args.max_tokens, temperature=0.6, seed=42 + i)
        for i in range(args.batch_size)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        del hf_model

        mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, model)

        print(f"Config: batch_size={args.batch_size}, prompt_len={prompt_len}, max_new_tokens={args.max_tokens}")
        param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
        print(f"Parameters: {param_count / 1e6:.1f}M")
        print()

        # Warmup (triggers JIT compilation)
        print(f"Warmup ({args.warmup} iterations)...")
        for i in range(args.warmup):
            t0 = time.perf_counter()
            result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
            jax.block_until_ready(result.generated_ids)
            t1 = time.perf_counter()
            print(f"  warmup {i}: {(t1 - t0) * 1000:.1f} ms")

        # Timed runs
        print(f"\nBenchmark ({args.runs} iterations)...")
        times = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
            jax.block_until_ready(result.generated_ids)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)
            tokens = sum(len(ids) for ids in result.generated_ids)
            print(f"  run {i}: {elapsed * 1000:.1f} ms, {tokens} tokens, {tokens / elapsed:.0f} tok/s")

        times = np.array(times)
        avg_tokens = sum(len(ids) for ids in result.generated_ids)
        print(f"\nResults:")
        print(f"  Mean: {times.mean() * 1000:.1f} ± {times.std() * 1000:.1f} ms")
        print(f"  Min:  {times.min() * 1000:.1f} ms")
        print(f"  Throughput: {avg_tokens / times.mean():.0f} tok/s")
        print(f"  Per-token:  {times.mean() / args.max_tokens * 1000:.2f} ms/tok")

        # Profiler trace
        if args.profile:
            print(f"\nCollecting profiler trace to {args.profile}...")
            jax.profiler.start_trace(args.profile)
            result = model.generate(input_ids, attention_mask, sampling_params=sampling_params)
            jax.block_until_ready(result.generated_ids)
            jax.profiler.stop_trace()
            print(f"Trace saved. Open with: chrome://tracing or https://ui.perfetto.dev/")
            print()
            print("What to look for:")
            print("  - Large 'memcpy DtoD' ops inside the while loop = buffer donation failure")
            print("  - Many small gaps between kernels = kernel launch overhead (try command buffers)")


if __name__ == "__main__":
    main()
