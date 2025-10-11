from __future__ import annotations

from enum import Enum
import os
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import torch
import torch.nn as nn
import safetensors.torch
from transformers import PretrainedConfig

from tx import models

if TYPE_CHECKING:
    pass


def get_dtype(dtype: str | torch.dtype) -> torch.dtype:
    "Convert dtype string to torch dtype."

    match str(dtype):
        case "torch.float32" | "float32":
            return torch.float32
        case "torch.bfloat16" | "bfloat16":
            return torch.bfloat16
        case "torch.float16" | "float16":
            return torch.float16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def get_model_class(config: PretrainedConfig) -> Callable[..., nn.Module]:
    "Get the correct model class based on the config."

    for architecture in config.architectures or []:
        if hasattr(models, architecture):
            return getattr(models, architecture)

    raise ValueError(f"None of the architectures {config.architectures} is currently supported.")


def get_param_key(path: str) -> str:
    "Get the safetensors key for a given model path."
    parts = path.split('.')
    if parts[-1] in {"weight"}:
        return path
    elif parts[-1] in {"lora_A", "lora_B"}:
        return path + ".weight"
    return path


def get_expert_key(path: str, expert_idx: int) -> str:
    "Get the safetensors key for an expert weight model path."
    parts = path.split('.')
    parts = [s if s != "weight" else f"experts.{expert_idx}.weight" for s in parts]
    return ".".join(parts)


def load_checkpoint(checkpoint_dir: str | os.PathLike, config: PretrainedConfig, model: nn.Module) -> None:
    tensors = {}
    for file in Path(checkpoint_dir).glob("*.safetensors"):
        tensors.update(safetensors.torch.load_file(file))

    state_dict = model.state_dict()
    updates = {}

    for name, param in state_dict.items():
        key = get_param_key(name)
        # Skip LoRA parameters that are not in the checkpoint
        if "lora_A" in name or "lora_B" in name or "lora_scaling" in name or "lora_ranks" in name:
            continue
        if "experts" in name and "weight" in name:
            # In order to load the expert weights, we stack the relevant tensors
            # HF stores each expert as Linear with shape [out, in]
            # We need [num_experts, in, out] for grouped_mm
            expert_tensors = [tensors[get_expert_key(name, i)].T for i in range(config.num_experts)]
            tensor_value = torch.stack(expert_tensors, dim=0)
        else:
            if key not in tensors:
                continue
            # HF and PyTorch both use [out, in] for Linear weights, so no transpose needed
            tensor_value = tensors[key]

        # Handle attention projection reshaping
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            tensor_value = tensor_value.reshape(param.shape)

        assert param.shape == tensor_value.shape, f"shape mismatch for {name}: {param.shape} != {tensor_value.shape}"
        updates[name] = tensor_value

    model.load_state_dict(updates, strict=False)


def save_checkpoint(config: PretrainedConfig, model: nn.Module, filename: str | os.PathLike) -> None:
    state_dict = model.state_dict()
    tensors = {}

    for name, param in state_dict.items():
        if "lora_scaling" in name or "lora_ranks" in name:
            # Skip LoRA config buffers, keep LoRA weights
            continue

        key = get_param_key(name)

        if "experts" in name and "weight" in name:
            # Save each expert weight separately
            for i in range(config.num_experts):
                tensors[get_expert_key(name, i)] = param[i, :, :]
            continue

        param_to_save = param
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            param_to_save = param.reshape(param.shape[0], -1)
        elif "o_proj" in name:
            param_to_save = param.reshape(-1, param.shape[-1])

        # HF and PyTorch both use [out, in] for Linear weights, so no transpose needed
        tensors[key] = param_to_save

    safetensors.torch.save_file(tensors, filename)


class OptimizerName(str, Enum):
    adamw = "adamw"
    sgd = "sgd"


def get_optimizer(optimizer_name: OptimizerName, optimizer_args: dict, parameters) -> torch.optim.Optimizer:
    match (optimizer_name, optimizer_args):
        case (OptimizerName.adamw, {"learning_rate": lr, **kwargs}):
            # Extract weight_decay if present, otherwise default to 0.0
            weight_decay = kwargs.pop("weight_decay", 0.0)
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
        case (OptimizerName.sgd, {"learning_rate": lr, **kwargs}):
            return torch.optim.SGD(parameters, lr=lr, **kwargs)
        case (_, {"learning_rate": _}):
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        case _:
            raise ValueError("The 'learning_rate' key must be provided in optimizer_args.")
