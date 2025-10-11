import torch
import torch.nn.functional as F
from transformers import AutoConfig
from huggingface_hub import snapshot_download

from tx.models import Qwen3ForCausalLM
from tx.utils.models import get_dtype, load_checkpoint
from tx.layers.lora import update_adapter_config


def test_lora_training():
    base_model = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(base_model)
    config.max_lora_adapters = 5
    config.max_lora_rank = 32

    checkpoint_path = snapshot_download(base_model, allow_patterns=["*.safetensors"])

    model = Qwen3ForCausalLM(config, dtype=get_dtype(config.dtype))
    load_checkpoint(checkpoint_path, config, model)

    # Set different ranks for each adapter (0: rank 16, 1: rank 8)
    update_adapter_config(model, adapter_index=0, lora_rank=16, lora_alpha=16)
    update_adapter_config(model, adapter_index=1, lora_rank=8, lora_alpha=8)

    # Create optimizer that only targets LoRA A and B parameters
    lora_params = [p for n, p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    batch = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype=torch.int32)
    target_ids = batch[:, 1:]
    input_ids = batch[:, :-1]
    adapter_indices = torch.tensor([0, 1], dtype=torch.int32)

    def loss_fn(model, input_ids, target_ids):
        outputs = model(input_ids, adapter_indices=adapter_indices)
        logits = outputs["logits"]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1).long(),
            reduction='mean'
        )

    # Helper to extract adapter params at specific index
    def get_adapter_params(model, adapter_idx):
        params = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                params[name] = param[adapter_idx].clone().detach()
        return params

    # Helper to extract out-of-rank params for an adapter
    def get_out_of_rank_params(model, adapter_idx, rank):
        params = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                params[name] = param[adapter_idx, :, rank:].clone().detach()
            elif 'lora_B' in name:
                params[name] = param[adapter_idx, rank:, :].clone().detach()
        return params

    # Save initial states
    initial_adapter_2_params = get_adapter_params(model, 2)
    initial_adapter_0_out_of_rank = get_out_of_rank_params(model, 0, 16)
    initial_adapter_1_out_of_rank = get_out_of_rank_params(model, 1, 8)

    # Training loop
    model.train()
    for step in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model, input_ids, target_ids)
        loss.backward()
        optimizer.step()

        print(f"Step {step}: loss = {float(loss):.4f}")

    def verify_params_unchanged(initial_params, final_params, error_msg_prefix):
        for name in initial_params:
            initial = initial_params[name]
            final = final_params[name]
            assert torch.allclose(initial, final), f"{error_msg_prefix} for {name}"

    # Verify adapter 2 (unused) was not modified
    final_adapter_2_params = get_adapter_params(model, 2)
    verify_params_unchanged(initial_adapter_2_params, final_adapter_2_params, "Adapter 2 was modified")

    # Verify out-of-rank params were not modified
    final_adapter_0_out_of_rank = get_out_of_rank_params(model, 0, 16)
    verify_params_unchanged(
        initial_adapter_0_out_of_rank, final_adapter_0_out_of_rank, "Adapter 0 out-of-rank params modified"
    )
    final_adapter_1_out_of_rank = get_out_of_rank_params(model, 1, 8)
    verify_params_unchanged(
        initial_adapter_1_out_of_rank, final_adapter_1_out_of_rank, "Adapter 1 out-of-rank params modified"
    )
