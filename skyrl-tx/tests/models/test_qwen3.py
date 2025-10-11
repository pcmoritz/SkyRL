import os
import tempfile

import numpy as np
from peft import LoraConfig, get_peft_model
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock

from tx.layers.lora import LoRAMixin
from tx.models import Qwen3ForCausalLM
from tx.models.qwen3 import Qwen3MoeSparseMoeBlock
from tx.utils.models import load_checkpoint


@pytest.mark.parametrize("tp", [1])  # TP=2 not supported in PyTorch version yet
def test_qwen3(tp: int):
    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", attn_implementation="eager", use_safetensors=True
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )

    # Save the HF model checkpoint so we can load our model from it
    with tempfile.TemporaryDirectory() as tmp:
        hf_model.save_pretrained(tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        model = Qwen3ForCausalLM(config, dtype=torch.float32)
        load_checkpoint(tmp, config, model)
        model.eval()

        with torch.no_grad():
            outputs = model(batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True)

        # Convert to numpy for comparison
        hf_hidden_0 = hf_outputs.hidden_states[0].numpy()
        hf_hidden_1 = hf_outputs.hidden_states[1].numpy()
        hf_hidden_last = hf_outputs.hidden_states[-1].numpy()

        our_hidden_0 = outputs["hidden_states"][0].numpy()
        our_hidden_1 = outputs["hidden_states"][1].numpy()
        our_hidden_last = outputs["hidden_states"][-1].numpy()

        assert np.allclose(hf_hidden_0, our_hidden_0, rtol=1e-6)
        assert np.allclose(hf_hidden_1, our_hidden_1, rtol=1e-3, atol=1e-3)
        assert np.allclose(hf_hidden_last, our_hidden_last, rtol=1e-3, atol=1e-3)


def load_moe_base_weights(torch_moe_layer: Qwen3MoeSparseMoeBlock, hf_moe_layer: HFQwen3MoeSparseMoeBlock) -> None:
    """Load base weights from HF MoE layer to PyTorch MoE layer."""
    with torch.no_grad():
        torch_moe_layer.gate.weight.copy_(hf_moe_layer.gate.weight)
        for i, expert in enumerate(hf_moe_layer.experts):
            torch_moe_layer.experts.gate_proj.weight[i, :, :] = expert.gate_proj.weight.T
            torch_moe_layer.experts.up_proj.weight[i, :, :] = expert.up_proj.weight.T
            torch_moe_layer.experts.down_proj.weight[i, :, :] = expert.down_proj.weight.T


def test_qwen3_moe_layer():
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    config = AutoConfig.from_pretrained(model_name)

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(4, 2, config.hidden_size)
    with torch.no_grad():
        hf_final_hidden_states, hf_router_logits = hf_moe_layer.forward(x)

    moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=torch.float32)
    load_moe_base_weights(moe_layer, hf_moe_layer)
    moe_layer.eval()

    with torch.no_grad():
        final_hidden_states, router_logits = moe_layer(x, return_router_logits=True)

    assert torch.allclose(hf_router_logits, router_logits, rtol=1e-4)
    assert torch.allclose(hf_final_hidden_states, final_hidden_states, rtol=1e-2, atol=1e-2)


def load_lora_weights(
    torch_module: LoRAMixin,
    adapter_idx: int,
    lora_A_weights: np.ndarray,
    lora_B_weights: np.ndarray,
    scaling: float,
    rank: int,
) -> None:
    """Load LoRA weights from numpy arrays to PyTorch module."""
    assert (
        torch_module.lora_A is not None
        and torch_module.lora_B is not None
        and torch_module.lora_scaling is not None
        and torch_module.lora_ranks is not None
    )
    with torch.no_grad():
        torch_module.lora_A[adapter_idx] = torch.from_numpy(lora_A_weights).to(torch_module.lora_A.dtype)
        torch_module.lora_B[adapter_idx] = torch.from_numpy(lora_B_weights).to(torch_module.lora_B.dtype)
        torch_module.lora_scaling[adapter_idx] = scaling
        torch_module.lora_ranks[adapter_idx] = rank


def test_qwen3_moe_layer_lora():
    """Test MoE LoRA by merging adapter into base weights and comparing outputs."""
    model_name = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", use_safetensors=True)
    config = AutoConfig.from_pretrained(model_name)

    # Enable LoRA
    config.max_lora_adapters = 3
    config.max_lora_rank = 4

    hf_moe_layer = hf_model.model.layers[0].mlp
    x = torch.randn(3, 4, config.hidden_size)

    moe_layer = Qwen3MoeSparseMoeBlock(config, dtype=torch.float32)
    load_moe_base_weights(moe_layer, hf_moe_layer)

    # Set LoRA weights for all adapters
    rng = np.random.default_rng(42)
    scaling = 2.0
    rank = config.max_lora_rank
    for adapter_idx in range(config.max_lora_adapters):
        for proj in [moe_layer.experts.gate_proj, moe_layer.experts.up_proj, moe_layer.experts.down_proj]:
            assert proj.lora_A is not None and proj.lora_B is not None
            lora_A = rng.normal(0, 1.0, proj.lora_A.shape[1:])
            lora_B = rng.normal(0, 1.0, proj.lora_B.shape[1:])
            load_lora_weights(proj, adapter_idx, lora_A, lora_B, scaling, rank)

    moe_layer.eval()
    # Test with different adapters per sample
    adapter_indices = torch.tensor([0, 2, 1], dtype=torch.int32)
    with torch.no_grad():
        output_with_lora, _ = moe_layer(x, adapter_indices=adapter_indices, return_router_logits=True)

    # Test each sample by comparing with merged weights for its adapter
    for sample_idx in range(len(adapter_indices)):
        adapter_idx = int(adapter_indices[sample_idx])

        # Create merged model by adding LoRA weights to base weights
        moe_layer_merged = Qwen3MoeSparseMoeBlock(config, dtype=torch.float32)
        with torch.no_grad():
            moe_layer_merged.gate.weight.copy_(moe_layer.gate.weight)

            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(moe_layer.experts, proj_name)
                proj_merged = getattr(moe_layer_merged.experts, proj_name)

                # For each expert, merge: base + scaling * (lora_A @ lora_B)
                for expert_idx in range(config.num_experts):
                    lora_A = proj.lora_A[adapter_idx, expert_idx, :, :]
                    lora_B = proj.lora_B[adapter_idx, expert_idx, :, :]
                    lora_delta = scaling * (lora_A @ lora_B)

                    merged_weight = proj.weight[expert_idx, :, :] + lora_delta
                    proj_merged.weight[expert_idx, :, :] = merged_weight

        moe_layer_merged.eval()
        # Run merged model on this sample
        x_sample = x[sample_idx : sample_idx + 1]
        with torch.no_grad():
            output_merged, _ = moe_layer_merged(x_sample, return_router_logits=True)

        assert torch.allclose(output_with_lora[sample_idx : sample_idx + 1], output_merged, rtol=1e-3, atol=1e-3)


def test_qwen3_lora():
    """Test multi-LoRA implementation by comparing with HuggingFace PEFT model using two different adapters."""
    base_model_name = "Qwen/Qwen3-0.6B"
    lora_adapters = ["charent/self_cognition_Alice", "charent/self_cognition_Bob"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Use two different inputs to test with different adapters
    inputs = ["The capital of France is", "My name is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)

    with tempfile.TemporaryDirectory() as base_tmp:
        base_hf_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, attn_implementation="eager", use_safetensors=True
        )
        base_hf_model.save_pretrained(base_tmp, safe_serialization=True)

        config = AutoConfig.from_pretrained(base_model_name)

        # Create HF models with different adapters
        hf_lora_models = []
        lora_configs = []
        for adapter_name in lora_adapters:
            lora_config = LoraConfig.from_pretrained(adapter_name)
            lora_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_configs.append(lora_config)

            hf_model = get_peft_model(
                AutoModelForCausalLM.from_pretrained(
                    base_model_name, attn_implementation="eager", use_safetensors=True
                ),
                lora_config,
            )
            hf_model.eval()
            hf_model.load_adapter(adapter_name, adapter_name="default")
            hf_lora_models.append(hf_model)

        config.max_lora_adapters = len(lora_adapters)
        config.max_lora_rank = max(cfg.r for cfg in lora_configs)

        model = Qwen3ForCausalLM(config, dtype=torch.float32)
        load_checkpoint(base_tmp, config, model)
        model.eval()

        # Get outputs from all HF models
        hf_outputs_list = []
        with torch.no_grad():
            for idx in range(len(lora_adapters)):
                hf_output = hf_lora_models[idx](
                    batch.input_ids[idx : idx + 1],
                    attention_mask=batch.attention_mask[idx : idx + 1],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hf_outputs_list.append(hf_output)

        # Load LoRA adapter weights from all adapters
        for i, layer in enumerate(model.model.layers):
            for adapter_idx, (hf_model, lora_config) in enumerate(zip(hf_lora_models, lora_configs)):
                hf_layer = hf_model.base_model.model.model.layers[i]
                for module, projections in [
                    ("mlp", ["gate_proj", "up_proj", "down_proj"]),
                    ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                ]:
                    for proj_name in projections:
                        hf_proj = getattr(getattr(hf_layer, module), proj_name)
                        load_lora_weights(
                            getattr(getattr(layer, module), proj_name),
                            adapter_idx=adapter_idx,
                            lora_A_weights=hf_proj.lora_A["default"].weight.detach().numpy().T,
                            lora_B_weights=hf_proj.lora_B["default"].weight.detach().numpy().T,
                            scaling=lora_config.lora_alpha / lora_config.r,
                            rank=lora_config.r,
                        )

        # Use different adapter indices for each input
        adapter_indices = torch.arange(len(lora_adapters), dtype=torch.int32)
        with torch.no_grad():
            outputs = model(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                output_hidden_states=True,
                adapter_indices=adapter_indices,
            )

        # Compare outputs with corresponding adapters
        for idx in range(len(lora_adapters)):
            assert torch.allclose(
                hf_outputs_list[idx].logits[0],
                outputs["logits"][idx],
                rtol=1e-3,
                atol=1e-3
            )
