import json
from pathlib import Path
import sys

from datasets import Dataset, load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
import typer

from tx.loaders import get_loader
from tx.utils.models import OptimizerName, get_dtype, get_model_class, get_optimizer, load_checkpoint, save_checkpoint
from tx.utils.log import ExperimentTracker, add_file_handler, get_tracker, logger

app = typer.Typer()


def loss_fn(model, batch):
    logits = model(batch["text"], attention_mask=batch["attention_mask"])["logits"]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch["target"].reshape(-1),
        reduction='mean'
    )
    return loss, logits


def train_step(model, optimizer: torch.optim.Optimizer, batch):
    optimizer.zero_grad()
    loss, logits = loss_fn(model, batch)

    loss.backward()

    # Compute gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    gradnorm = total_norm ** 0.5

    optimizer.step()
    return loss, gradnorm


def train(
    model_name: str = typer.Option(..., "--model", help="HuggingFace model ID or local model path"),
    dataset: str = typer.Option(..., "--dataset", help="HuggingFace dataset to use for training"),
    loader_name: str = typer.Option("tx.loaders.text", "--loader", help="Loader used for loading the dataset"),
    split: str = typer.Option("train", "--split", help="The dataset split to use"),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="The output directory where the model predictions and checkpoints will be written"
    ),
    load_checkpoint_path: Path | None = typer.Option(
        None, "--load-checkpoint-path", help="If specified, resume training from this checkpoint"
    ),
    save_steps: int = typer.Option(500, "--save-steps", help="Number of steps between checkpoints"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="The maximum number of training steps"),
    batch_size: int = typer.Option(..., "--batch-size", help="Batch size of each training batch"),
    optimizer_name: OptimizerName = typer.Option("adamw", "--optimizer", help="Which optimizer to use"),
    optimizer_args: dict = typer.Option(
        '{"learning_rate": 1e-5, "weight_decay": 0.1}',
        "--optimizer-args",
        help="Arguments for the optimizer (in JSON format)",
        parser=json.loads,
    ),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", help="Device to use for training"),
    tracker_name: ExperimentTracker | None = typer.Option(
        None, "--tracker", help="Experiment tracker to report results to"
    ),
    tracker_args: dict = typer.Option(
        "{}",
        "--tracker-args",
        help="Arguments that will be passed to the experiment tracker (in JSON format)",
        parser=json.loads,
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_dir / "tx.log")
    logger.info(f"tx was invoked with 'tx {' '.join(sys.argv[1:])}'")

    train_dataset = load_dataset(dataset, split=split)
    assert isinstance(train_dataset, Dataset)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tracker = get_tracker(tracker_name, config, **tracker_args)
    loader = get_loader(loader_name)

    model_class = get_model_class(config)
    device_obj = torch.device(device)

    # Set manual seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    model = model_class(config, dtype=get_dtype(config.dtype), device=device_obj)
    model = model.to(device_obj)
    optimizer = get_optimizer(optimizer_name, optimizer_args, model.parameters())

    if load_checkpoint_path:
        load_checkpoint(load_checkpoint_path, config, model)

    num_steps = train_dataset.num_rows / batch_size
    for step, (batch, metrics) in enumerate(loader(tokenizer, train_dataset, batch_size)):
        if max_steps and step >= max_steps:
            break

        # Move batch to device
        batch = {k: v.to(device_obj) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model.train()
        loss, gradnorm = train_step(model, optimizer, batch)
        tracker.log({"epoch": step / num_steps, **metrics, "gradnorm": gradnorm, "loss": loss.item()}, step)

        if step % save_steps == 0:
            logger.info(f"Saving checkpoint to {output_dir}")
            save_checkpoint(config, model, output_dir / "model.safetensors")

    logger.info(f"Saving final checkpoint to {output_dir}")
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_checkpoint(config, model, output_dir / "model.safetensors")
