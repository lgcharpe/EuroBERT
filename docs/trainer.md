# **Optimus Training Library**

Optimus is the EuroBERT training library compatible with CPU, AMD, or NVIDIA hardware! This repository provides a **flexible and scalable training environment** with fully customizable model, data, and training parameters, supporting [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and [Flash Attention](https://github.com/Dao-AILab/flash-attention). 

Optimus is designed to allow **resumable training**, whether you're using the same or a different hardware configuration. It also supports **Fully Sharded Data Parallel (FSDP)**, **Distributed Data Parallel (DDP)**, and other parallelism strategies, enabling efficient scaling across multiple GPUs or nodes.

Whether you're a beginner or an expert, let's get started!

## 📑 **Table of Contents**
- 🚀 [**Quick Start**](#quick-start)
- ⚙️ [**Configuration**](#configuration)
  - 🏋️ [Training Configuration](#training-configuration)
  - 📊 [Data Configuration](#data-configuration)
  - 🤖 [Model Configuration](#model-configuration)

## 🚀 **Quick Start**

Ready to train EuroBERT ?
To install the Optimus training library simply run:  
```bash
pip install git+https://github.com/Nicolas-BZRD/EuroBERT.git
```
or, for development purposes, clone the repository and install it in editable mode:
```bash
git clone https://github.com/Nicolas-BZRD/EuroBERT.git
cd EuroBERT
pip install -e .
```

You can then launch training with the `python -m optimus.train` package. We additionally provide a [complete tutorial for continuous training of EuroBERT](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/examples/continuous_pretraining.ipynb) to help practitioners with their first training. For extensive training requiring further optimization, feel free to reach us at `nicolas(dot)boizard[at]centralesupelec(dot)fr`.

## ⚙️ **Configuration**

Customize your training by passing parameters via the command line or config file. Below are the details for each configuration section.

### 🏋️ **Training Configuration**

Fine-tune your training process with these parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| project_name | str | training | Project name for TensorBoard logging. |
| reload_checkpoint | str | None | Path to a checkpoint to resume training. |
| output_dir | str | output | Directory to save training outputs. |
| lr | float | 1e-4 | Learning rate. |
| num_epochs | int | 1 | Number of training epochs. |
| clip_grad_norm | float | 1.0 | Clip gradient norm. |
| gradient_accumulation_steps | int | 1 | Number of steps to accumulate gradients before updating the model. |
| weight_decay | float | 0.1 | Weight decay. |
| beta1 | float | 0.9 | Beta1 for Adam optimizer. |
| beta2 | float | 0.95 | Beta2 for Adam optimizer. |
| eps | float | 1e-5 | Epsilon for Adam optimizer. |
| fused | bool | False | Use fused optimizer which uses a single kernel. |
| lr_scheduler | str | WarmupStableDecayLR | LR scheduler (`WarmupStableDecayLR`, `CosineAnnealingLR` or `OneCycleLR`). |
| pct_start | float | 0.01 | Percentage of iterations for increasing the learning rate. |
| div_factor | int | 0 | Initial divisor for scheduler, if 0, the initial learning rate is 0. |
| end_start | float | 0 | Percentage of iterations for decreasing the learning rate(if `WarmupStableDecayLR` and 'end_start==1`no decay). |
| final_div_factor | int | 0 | Final divisor for scheduler, if 0, the final learning rate is 0. |
| compile_model | bool | False | Compile model. |
| compile_mode | str | None | Compilation mode. |
| compile_options | dict | None | Compilation options. |
| run_validation | bool | True | Run validation during training. |
| validation_step | int | 5000 | Run validation every `validation_step` iterations. |
| save_step | int | 5000 | Save model every `save_step` iterations. |
| save_model | bool | True | Save model during training. |
| save_optimizer | bool | True | Save optimizer state with model. |
| save_scheduler | bool | True | Save scheduler state with model. |
| save_data_loader | bool | True | Save data loader state with model. |
| save_config | bool | True | Save configuration with model. |
| mlm_probability | float | 0.3 | Probability of masking a token. |
| mask_probability | float | 1.0 | Probability of replacing a masked token with the mask token. |
| random_probability | float | 0.0 | Probability of replacing a masked token with a random token. |
| original_probability | float | 0.0 | Probability of keeping the original token. |
| skip_reload_scheduler | bool | False | Skip reloading the scheduler. |
| skip_reload_dataloader | bool | False | Skip reloading the data loader. |
| skip_reload_tensorboard | bool | False | Skip reloading the tensorboard. |
| fsdp | bool | False | Enable FullyShardedDataParallel (FSDP). |
| ddp | bool | False | Enable DistributedDataParallel (DDP). |
| mixed_bfloat16 | bool | True | Enable mixed precision training for regular and ddp training. |
| _mixed_precision | str | bfloat16 | FSDP training ShardingStrategy (`float32`, `float16`, `bfloat16`, `mixed_float16`, `mixed_bfloat16`, `bfloat16_reduce_32`), [PyTorch doc](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision), [Config file](https://github.com/Nicolas-BZRD/EuroBERT/blob/main/optimus/trainer/configuration/distributed.py).|
| seed | int | 42 | Random seed for reproducibility. |
| tensorboard | bool | True | Enable tensorboard logging. |
| profile | bool | False | Enable profiling. |
| exit_end_profiling | bool | True | Exit after profiling. |
| profiler_output | str | chrome | Type for the profiler output (chrome or tensorboard). |
| log_every_n_steps | int | 10 | Log every `log_every_n_steps` iterations. |

### 📊 **Data Configuration**

Control how your data is processed and fed into the model with these options.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data_mix_path | str | ./exemples/mix | Path to the data mix folder containing `train.json` and optionally `eval.json`. This JSON file contains parameter configurations passed to create a dataset with similar configuration possibilities as [MosaicML](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/mixing_data_sources.html). |
| shuffle | bool | True | Shuffle the dataset. |
| batch_size | int | 12 | Number of samples per batch. |
| prefetch_factor | int | 1 | Batches to prefetch for efficiency. |
| num_workers | int | 1 | Number of worker processes for data loading. |
| predownload | int | 1 | Files to predownload. |
| length | int | 2048 | Maximum sentence length. |
| var_len | bool | False | Enable variable-length sentences. |
| num_canonical_nodes | int | 0 | Number of canonical nodes. |
| pin_memory | bool | True | Pin memory for faster GPU transfer. |
| step_to_skip | int | 0 | Steps to skip during training. |
| seed | int | 42 | Random seed for reproducibility. |

### 🤖 **Model Configuration**

This section defines your model's architecture. If you provide a huggingface_id, other parameters are locked. Otherwise, for custom models, unspecified parameters will use defaults configuration of the [model choosed](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/optimus/trainer/model/encoder).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| huggingface_id | str | None | Hugging Face model ID (overrides other parameters). |
| tokenizer_path_or_name | str | meta-llama/Meta-Llama-3-8B-Instruct | Path or name of the tokenizer. |
| mask_token_id | int | 128002 | ID of the mask token. |
| gpu | bool | True | Move model on GPU if available. |
| model_name | str | bert | Model type (bert or eurobert, [here](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/optimus/trainer/model/encoder)). |
| model_size | str | 3b | Model size (e.g., 210m, 310m, 2b). |
| vocab_size | int | 128256 | Vocabulary size. |
| embedding_size | int | None | Embedding size (defaults to model size if unset). |
| num_head | int | None | Number of attention heads. |
| num_kv_head | int | None | Number of key-value heads. |
| num_layer | int | None | Number of layers. |
| block_size | int | None | Block size for processing. |
| dropout | float | None | Dropout probability. |
| mlp_hidden_dim | int | None | Hidden dimension of the MLP. |
| bias | bool | None | Use bias in layers. |
| attn_impl | str | None | Attention implementation. |
| rope_base | int | None | Base scaling factor for ROPE. |
| fused_rms_norm | bool | False | Use fused RMS normalization. |
| fused_rope | bool | False | Use fused RoPE embeddings. |
| fused_swiglu | bool | False | Use fused SwiGLU activation. |
| fused_cross_entropy | bool | False | Use fused cross entropy. |
| fused_linear_cross_entropy | bool | False | Use fused linear cross entropy. |

Pro Tip: Want to create a custom model? Check out the example in [optimus/trainer/model/encoder/eurobert.py](https://github.com/Nicolas-BZRD/EuroBERT/tree/main/optimus/trainer/model/encoder/eurobert.py) within this repository!

---

## 🏃 **Running Training**

### Single GPU

```bash
python -m optimus.train \
    --model_name eurobert --model_size 210m \
    --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_mix_path ./data/mix \
    --output_dir ./runs --project_name my_run \
    --lr 1e-4 --num_epochs 1 --batch_size 8 --length 2048
```

### Multi-GPU: DDP

```bash
torchrun --nproc_per_node=8 -m optimus.train \
    --ddp True \
    --model_name eurobert --model_size 610m \
    --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_mix_path ./data/mix \
    --output_dir ./runs --project_name ddp_run \
    --lr 2e-4 --num_epochs 1 --batch_size 8
```

### Multi-GPU: FSDP

```bash
torchrun --nproc_per_node=8 -m optimus.train \
    --fsdp True \
    --model_name eurobert --model_size 2b \
    --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_mix_path ./data/mix \
    --output_dir ./runs --project_name fsdp_run \
    --lr 1e-4 --num_epochs 1 --batch_size 4 --gradient_accumulation_steps 4
```

### Multi-node: FSDP (2 nodes × 8 GPUs)

```bash
torchrun \
    --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=job123 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    -m optimus.train \
    --fsdp True --model_name eurobert --model_size 2b \
    --data_mix_path ./data/mix ...
```

### Resume from Checkpoint

```bash
python -m optimus.train \
    --reload_checkpoint ./runs/my_run/checkpoints/5000 \
    --data_mix_path ./data/mix
```

The configuration is automatically restored from `checkpoints/5000/config.json`. Only parameters you explicitly pass on the CLI will override the saved config.

### Use a HuggingFace Model

Pass a HuggingFace model ID instead of a model name. When `huggingface_id` is set, all `model_*` architecture parameters are ignored and the model is loaded with `AutoModelForMaskedLM`.

```bash
python -m optimus.train \
    --huggingface_id "Nicolas-BZRD/EuroBERT-210m" \
    --data_mix_path ./data/mix \
    --lr 5e-5 --num_epochs 1 --batch_size 8
```

---

## 📁 **Checkpoint Layout**

After training, checkpoints are saved under:

```
output_dir/
└── <project_name>/
    ├── tensorboard/                  # TensorBoard event files
    ├── profiler/                     # Profiler traces (if enabled)
    └── checkpoints/
        └── <step>/
            ├── model.pt              # Model state dict
            ├── optimizer.pt          # Optimizer state dict
            ├── scheduler.pt          # LR scheduler state dict
            ├── train_dataloader.pt   # Dataloader position (for exact resume)
            └── config.json           # Full serialised Config
```

You can disable saving any component separately:

```bash
python -m optimus.train ... \
    --save_optimizer False \
    --save_scheduler False \
    --save_data_loader False
```

---

## 📈 **Learning Rate Schedulers**

Three schedulers are built in, selected via `--lr_scheduler`:

### `WarmupStableDecayLR` (default)

Three-phase schedule: linear warm-up → constant plateau → linear decay.

| Parameter | Meaning |
|---|---|
| `pct_start` | Warm-up length (≤ 1 = fraction of total steps, > 1 = absolute steps) |
| `div_factor` | `initial_lr = max_lr / div_factor` (0 = start from 0) |
| `end_start` | Decay length (same units as `pct_start`) |
| `final_div_factor` | `final_lr = max_lr / final_div_factor` (0 = decay to 0) |

Example — 1 % warm-up, hold, decay the last 10 %:
```bash
--lr_scheduler WarmupStableDecayLR --lr 1e-4 \
--pct_start 0.01 --div_factor 100 \
--end_start 0.10 --final_div_factor 100
```

### `CosineAnnealingLR`

Standard cosine decay over the total number of steps. No extra parameters needed.

### `OneCycleLR`

Uses `pct_start`, `div_factor`, and `final_div_factor` — see [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html).

---

## 🔩 **Kernel Optimisations**

Optional fused kernels require the `kernel` extra install:

```bash
pip install "git+https://github.com/Nicolas-BZRD/EuroBERT.git[kernel]"
# installs flash_attn and liger_kernel
```

| Flag | Kernel | Benefit |
|---|---|---|
| `--attn_impl flash` | Flash Attention 2 | Faster + lower-memory attention (not compatible with `torch.compile`) |
| `--fused_rms_norm True` | Liger RMSNorm | ~30 % faster normalisation |
| `--fused_rope True` | Liger RoPE | ~25 % faster positional embedding |
| `--fused_swiglu True` | Liger SwiGLU | ~20 % faster FFN activation |
| `--fused_cross_entropy True` | Liger CrossEntropy | Memory-efficient loss (avoids materialising large logit tensors) |

`torch.compile` (`--compile_model True`) is compatible with all fused kernels **except** Flash Attention. When using both `flash` attention and `torch.compile`, Flash SDP is automatically disabled and the math SDPA kernel is used instead.

---

## 🔍 **Profiling**

Enable PyTorch profiler with:

```bash
python -m optimus.train ... \
    --profile True \
    --profiler_output chrome \   # or "tensorboard"
    --exit_end_profiling True    # stop after 20 steps
```

Profiler output is saved to `output_dir/<project_name>/profiler/`. Open Chrome traces at `chrome://tracing`.

---

## 📊 **TensorBoard**

TensorBoard logging is enabled by default. Tracked metrics:

- `Loss/train` — training loss per step
- `Loss/eval` — validation loss (if eval set provided)
- `Gradient norm` — grad norm after clipping
- `Learning rate` — current LR from scheduler
- `Time/step in seconde` — wall-clock seconds per step
- `Tokens seen` — cumulative tokens processed
- `Tokens seen/second` — throughput

```bash
tensorboard --logdir ./runs/my_run/tensorboard
```
