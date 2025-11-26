"""Configuration classes for models with LoRA support."""

from transformers import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    """Qwen3 configuration for tx.

    Wraps a HuggingFace PretrainedConfig with additional parameters
    for Multi-LoRA training and tensor parallelism.

    Args:
        config: A HuggingFace PretrainedConfig object (e.g., from Qwen3Config.from_pretrained())
        max_lora_adapters: Maximum number of concurrent LoRA adapters
        max_lora_rank: Maximum rank for LoRA adapters
        shard_attention_heads: Whether to shard attention across tensor parallel devices
    """

    # Type hints for LoRA attributes
    max_lora_adapters: int
    max_lora_rank: int
    shard_attention_heads: bool

    def __init__(
        self, config: PretrainedConfig, *, max_lora_adapters: int, max_lora_rank: int, shard_attention_heads: bool
    ):
        # Copy all attributes from the base config
        super().__init__(**config.to_dict())

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads

    def __repr__(self):
        """Custom repr to avoid instantiation issues in to_diff_dict."""
        config_str = super().__repr__()
        return f"{self.__class__.__name__}(config={config_str}, max_lora_adapters={self.max_lora_adapters}, max_lora_rank={self.max_lora_rank}, shard_attention_heads={self.shard_attention_heads})"
