"""Test that prefill logits match between HF and TX."""

import gc
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from tx.models.configs import Qwen3Config
from tx.models.qwen3 import Qwen3ForCausalLM
from tx.utils.models import load_safetensors


MODEL_NAME = "Qwen/Qwen3-4B"


def test_prefill_logits_match():
    """Compare prefill logits between HF and TX."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    prompt = "The capital of France is"
    tokens = tokenizer.encode(prompt, add_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {tokens}")

    # Load HuggingFace model
    print("\n=== HuggingFace ===")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    hf_model.eval()

    input_ids_torch = torch.tensor([tokens]).to(hf_model.device)

    with torch.no_grad():
        # Compare embeddings
        hf_embed = hf_model.model.embed_tokens(input_ids_torch)
        print(f"HF embed shape: {hf_embed.shape}")
        print(f"HF embed[0,0,:5]: {hf_embed[0, 0, :5]}")

        hf_output = hf_model(input_ids_torch)
        hf_logits = hf_output.logits[0, -1].float().cpu().numpy()  # Last position

    hf_top10_idx = np.argsort(hf_logits)[-10:][::-1]
    print(f"HF top10 tokens: {hf_top10_idx}")
    print(f"HF top10 logits: {hf_logits[hf_top10_idx]}")
    print(f"HF argmax: {hf_top10_idx[0]} ({tokenizer.decode([hf_top10_idx[0]])!r})")

    # Save for TX
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        # Load TX model
        print("\n=== TX ===")
        base_config = PretrainedConfig.from_pretrained(MODEL_NAME)
        config = Qwen3Config(
            base_config, max_lora_adapters=32, max_lora_rank=32, shard_attention_heads=True
        )

        mesh = jax.make_mesh((1, 4), ("dp", "tp"))
        with jax.set_mesh(mesh):
            tx_model = Qwen3ForCausalLM(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
        load_safetensors(tmp, config, tx_model)

        input_ids_jax = jnp.array([tokens])
        attention_mask_jax = jnp.ones_like(input_ids_jax)
        positions = jnp.arange(len(tokens))[None, :]

        # Compare embeddings first
        tx_embed = tx_model.model.embed_tokens(input_ids_jax)
        print(f"TX embed shape: {tx_embed.shape}")
        print(f"TX embed[0,0,:5]: {tx_embed[0, 0, :5]}")

        tx_output = tx_model(input_ids_jax, attention_mask=attention_mask_jax, positions=positions)
        tx_logits = np.array(tx_output.logits[0, -1].astype(jnp.float32))

        tx_top10_idx = np.argsort(tx_logits)[-10:][::-1]
        print(f"TX top10 tokens: {tx_top10_idx}")
        print(f"TX top10 logits: {tx_logits[tx_top10_idx]}")
        print(f"TX argmax: {tx_top10_idx[0]} ({tokenizer.decode([tx_top10_idx[0]])!r})")

        # Compare
        print("\n=== Comparison ===")
        print(f"Argmax match: {hf_top10_idx[0] == tx_top10_idx[0]}")

        # Check logit differences for top tokens
        print("\nLogit differences for HF top10:")
        for idx in hf_top10_idx:
            diff = abs(hf_logits[idx] - tx_logits[idx])
            print(f"  Token {idx} ({tokenizer.decode([idx])!r}): HF={hf_logits[idx]:.4f}, TX={tx_logits[idx]:.4f}, diff={diff:.4f}")


if __name__ == "__main__":
    test_prefill_logits_match()
