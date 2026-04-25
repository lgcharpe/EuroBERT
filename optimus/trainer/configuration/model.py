from dataclasses import dataclass


@dataclass
class ModelConfig:
    huggingface_id: str = (
        None  # Huggingface model id (if provided, other model parameters are ignored)
    )
    tokenizer_path_or_name: str = "/scratch /project_465002893/nb-embed/EuroBERT/Australis"  # Path or name of the tokenizer
    mask_token_id: int = 2
    gpu: bool = True

    # Model parameters
    model_name: str = "eurobert"
    model_size: str | None = "210m"

    # If parameters are not provided, default values of the model size are used.
    vocab_size: int | None = None  # Vocabulary size
    embedding_size: int | None = None  # Embedding size
    num_head: int | None = None  # Number of heads
    num_kv_head: int | None = None  # Number of key-value heads
    num_layer: int | None = None  # Number of layers
    block_size: int | None = None  # Block size
    dropout: int | None = None  # Dropout
    mlp_hidden_dim: int | None = None  # MLP hidden dimension
    bias: bool | None = None  # Bias
    attn_impl: str | None = None  # Attention implementation
    rope_base: int | None = None  # Rope base

    fused_rms_norm: bool = None  # Fused RMS norm
    fused_rope: bool = None  # Fused rope
    fused_swiglu: bool = None  # Fused swiglu
    fused_cross_entropy: bool = None  # Fused cross entropy
    fused_linear_cross_entropy: bool = None  # Fused linear cross entropy
