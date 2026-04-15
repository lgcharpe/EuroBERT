# Codebase Architecture

This document describes the overall structure of the Optimus codebase and how all of its components relate to each other.

## Table of Contents

- [Codebase Architecture](#codebase-architecture)
  - [Table of Contents](#table-of-contents)
  - [Repository Layout](#repository-layout)
  - [Data Flow Overview](#data-flow-overview)
  - [Module Breakdown](#module-breakdown)
    - [`optimus/train.py`](#optimustrainpy)
    - [`optimus/trainer/`](#optimustrainer)
      - [`configuration/`](#configuration)
      - [`model/`](#model)
      - [`data.py`](#datapy)
      - [`pretrain.py`](#pretrainpy)
      - [`distributed.py`](#distributedpy)
      - [`script/warmup_stable_decay_lr.py`](#scriptwarmup_stable_decay_lrpy)
    - [`optimus/dataprocess/`](#optimusdataprocess)
    - [`optimus/tokenizer/`](#optimustokenizer)

---

## Repository Layout

```
EuroBERT/
├── optimus/
│   ├── train.py                        # Entry point: ties everything together
│   ├── trainer/
│   │   ├── pretrain.py                 # Training loop, checkpointing, evaluation
│   │   ├── data.py                     # Dataset + dataloader creation, MLM masking
│   │   ├── distributed.py              # DDP / FSDP setup
│   │   ├── configuration/
│   │   │   ├── configs.py              # Top-level Config object (aggregates all sub-configs)
│   │   │   ├── train.py                # TrainConfig dataclass
│   │   │   ├── dataset.py              # DatasetConfig dataclass
│   │   │   ├── model.py                # ModelConfig dataclass
│   │   │   ├── distributed.py          # DistributedConfig (sharding, mixed precision)
│   │   │   └── system.py              # SystemConfig (rank, world_size — auto-filled)
│   │   ├── model/
│   │   │   ├── load.py                 # load_model(), load_tokenizer(), compile_model()
│   │   │   ├── model.py                # Shared building blocks: TransformerEncoder, Block, Attention, MLP, RoPE, RMSNorm ...
│   │   │   ├── tools.py                # ModelTools helpers (summary, GPU cache clearing)
│   │   │   └── encoder/
│   │   │       ├── eurobert.py         # EuroBERT architecture (210m / 610m / 2b)
│   │   │       └── bert.py             # Classical BERT architecture (280m / 3b)
│   │   └── script/
│   │       ├── cache.py                # KV-cache used during inference
│   │       └── warmup_stable_decay_lr.py  # Custom Warmup → Stable → Decay LR scheduler
│   ├── dataprocess/
│   │   ├── tokenize_dataset.py         # Tokenize raw text files → MDS shards
│   │   ├── pack_dataset.py             # Pack tokenized tokens into fixed-size blocks
│   │   ├── subsample_dataset.py        # Split large MDS datasets into sub-indexes
│   │   ├── inspect_dataset.py          # Print decoded samples from an MDS dataset
│   │   └── dataset/                    # One file per supported raw dataset format
│   │       ├── fineweb.py
│   │       ├── wikipedia.py
│   │       └── ...                     # (17 total — see dataprocess.md)
│   └── tokenizer/
│       ├── train_from_scratch.py       # Train a BPE tokenizer from a text corpus
│       ├── train_from_old.py           # Retrain/extend an existing tokenizer
│       └── convert_to_hf.py           # Convert tokenizer.json → AutoTokenizer-compatible dir
├── docs/
│   ├── architecture.md                 # This file
│   ├── dataprocess.md                  # Data pipeline reference
│   ├── trainer.md                      # Trainer configuration reference
│   ├── tokenizer.md                    # Tokenizer training reference
│   └── extending.md                    # How to extend the codebase
└── examples/
    └── continuous_pretraining.ipynb    # End-to-end tutorial notebook
```

---

## Data Flow Overview

The full pipeline, from raw text to a trained model, consists of four stages:

```
Raw text files (parquet / jsonl / txt)
        │
        ▼
[1] optimus/dataprocess/tokenize_dataset.py
        │  Uses a dataset adapter (optimus/dataprocess/dataset/<name>.py)
        │  to stream text → tokenizes with AutoTokenizer or tiktoken
        │  → writes MosaicML Streaming (MDS) shards
        ▼
MDS shards (tokens as int32 numpy arrays + metadata JSON)
        │
        ▼ (optional)
[2] optimus/dataprocess/pack_dataset.py
        │  Packs individual token arrays into fixed block_size chunks
        │  → writes new MDS shards partitioned into train/ and val/
        ▼
Packed MDS shards (block_size tokens per record)
        │
        ▼ (optional)
[3] optimus/dataprocess/subsample_dataset.py
        │  Splits a large MDS directory into sub-indexes for
        │  canonical-node-based distributed loading
        ▼
Sub-indexed MDS shards
        │
        ▼
[4] optimus/train.py
        │  Config → load_model() + load_tokenizer()
        │  → Data (MaskingDataset + StreamingDataLoader)
        │  → Pretrain.train() (MLM training loop)
        ▼
Checkpoints: model.pt, optimizer.pt, scheduler.pt,
             train_dataloader.pt, config.json
```

---

## Module Breakdown

### `optimus/train.py`

The top-level entry point. Accepts all configuration parameters as CLI arguments via `fire`. Its responsibility is minimal: orchestrate the other modules.

1. Build a `Config` object from CLI kwargs.
2. Optionally initialise `Distributed` (DDP or FSDP).
3. Call `load_model()` and `load_tokenizer()`.
4. Wrap the model for distributed training if needed.
5. Patch `mosaicml_streaming` spanner (fixes a known streaming bug).
6. Create a `Data` object which builds `StreamingDataset` and `StreamingDataLoader`.
7. Create a `Pretrain` object and call `.train()`.

**How to run:**

```bash
# Single GPU
python -m optimus.train \
    --model_name eurobert --model_size 210m \
    --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_mix_path ./data/mix \
    --output_dir ./runs --project_name my_run \
    --lr 1e-4 --num_epochs 1 --batch_size 8

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 -m optimus.train \
    --ddp True --model_name eurobert --model_size 610m \
    --data_mix_path ./data/mix ...

# Multi-GPU (FSDP)
torchrun --nproc_per_node=8 -m optimus.train \
    --fsdp True --model_name eurobert --model_size 2b \
    --data_mix_path ./data/mix ...

# Multi-node (FSDP, 2 nodes × 8 GPUs)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m optimus.train --fsdp True ...
```

---

### `optimus/trainer/`

#### `configuration/`

All configuration is represented as Python dataclasses. The master `Config` class (in `configs.py`) aggregates five sub-configs:

| Sub-config | Dataclass | Key responsibility |
| --- | --- | --- |
| `config.train` | `TrainConfig` | LR, optimizer, grad accum, masking, checkpointing |
| `config.data` | `DatasetConfig` | Batch size, sequence length, data-mix path |
| `config.model` | `ModelConfig` | Architecture name/size, tokenizer path, fused kernels |
| `config.distributed` | `DistributedConfig` | FSDP sharding strategy, mixed-precision dtype |
| `config.system` | `SystemConfig` | Rank, world_size (auto-populated, do not set manually) |

Parameters can be passed as CLI flags to `optimus.train` or via a `config.json` checkpoint for resuming. Every `key=value` pair in the CLI is routed to whichever sub-config owns a field with that name, so only one flat namespace is needed.

Saving config: `config.save(path)` serialises all five sub-configs into `path/config.json`.

#### `model/`

- **`model.py`** — All reusable PyTorch building blocks:
  - `TransformerEncoder` — Base class used by both EuroBERT and Bert. Holds embedding, `nn.ModuleList` of `Block`s, final layer-norm, and lm_head. Handles the forward pass and cross-entropy loss.
  - `Block` — Pre-norm residual block (LayerNorm/RMSNorm → Attention → residual + LayerNorm/RMSNorm → MLP → residual).
  - `SelfAttention` (abstract) + `TorchSelfAttention` / `FlashSelfAttention` — GQA-capable self-attention.
  - `RoPE` — Rotary positional embedding.
  - `RMSNorm` — Root-mean-square layer normalisation.
  - `SwigluMLP` / `GeluMLP` — Feed-forward networks.
  - `CustomEmbedding` — `torch.compile`-friendly embedding.

- **`encoder/eurobert.py`** — EuroBERT: RoPE + RMSNorm + GQA + SwiGLU, three size presets (210m / 610m / 2b).
- **`encoder/bert.py`** — Classical BERT: learned positional embeddings, LayerNorm, GeLU MLP, two size presets (280m / 3b).
- **`load.py`** — `load_model()` dispatches to the right encoder class (or `AutoModelForMaskedLM` for HuggingFace models). `load_tokenizer()` loads via `AutoTokenizer`. `compile_model()` wraps with `torch.compile`.

#### `data.py`

`Data` creates the `MaskingDataset` (a `StreamingDataset` subclass) and `StreamingDataLoader`. The dataset reads packed MDS shards, applies MLM masking on-the-fly according to:

- `mlm_probability` — fraction of tokens masked in each sequence
- `mask_probability` — of those, fraction replaced with `[MASK]`
- `random_probability` — fraction replaced with a random token
- `original_probability` — fraction kept as-is

Two collate functions are available:

- `to_torch_collate_fn` — fixed-length sequences (all same length)
- `to_torch_collate_var_len_fn` — variable-length, produces `cu_seq_lens` for packed-sequence Flash Attention

The data mix is defined by a JSON file at `data_mix_path/train.json`, which is a list of MosaicML `Stream` configs (local/remote paths, proportions, etc.).

#### `pretrain.py`

`Pretrain` owns the training loop:

- `__init__`: creates the AdamW optimizer, calls `get_scheduler()`, optionally resumes from checkpoint, optionally `torch.compile`s the model, initialises KV-cache.
- `train()`: iterates over the dataloader with gradient accumulation, mixed-precision autocast, grad norm clipping, periodic logging (stdout + TensorBoard), periodic validation (`.eval()`), periodic checkpointing (`.save()`), and optional profiling.
- `save()`: stores `model.pt`, `optimizer.pt`, `scheduler.pt`, `train_dataloader.pt`, and `config.json` under `output_dir/<project_name>/checkpoints/<step>/`.
- `resume()`: reloads all of the above from a given checkpoint path.
- `get_scheduler()`: dispatches to `WarmupStableDecayLR`, `CosineAnnealingLR`, or `OneCycleLR`.

#### `distributed.py`

`Distributed` wraps the PyTorch distributed primitives:

- Initialises the NCCL process group from `LOCAL_RANK` / `RANK` / `WORLD_SIZE` environment variables (set by `torchrun`).
- `fsdp_setup_model()` — wraps the model in `FullyShardedDataParallel`.
- `ddp_setup_model()` — wraps the model in `DistributedDataParallel`.
- `save_fsdp_model_optimizer()` / `load_fsdp_model_optimizer()` — FSDP-aware checkpoint I/O using `torch.distributed.checkpoint`.

#### `script/warmup_stable_decay_lr.py`

`WarmupStableDecayLR` implements a three-phase schedule:

1. **Warm-up**: linear ramp from `max_lr / div_factor` → `max_lr` over `pct_start` iterations.
2. **Stable**: constant `max_lr`.
3. **Decay**: linear ramp from `max_lr` → `max_lr / final_div_factor` over `end_start` iterations.

All phase lengths can be given as absolute step counts (> 1) or as fractions of total training steps (≤ 1).

---

### `optimus/dataprocess/`

See [dataprocess.md](dataprocess.md) for the full reference. In summary:

| Script | Input | Output |
| --- | --- | --- |
| `tokenize_dataset.py` | Raw files (parquet/jsonl/…) | MDS shards of `{tokens: int32[], metadata: json}` |
| `pack_dataset.py` | MDS shards (any token length) | MDS shards of exactly `block_size` tokens |
| `subsample_dataset.py` | One MDS directory | Same directory re-indexed into sub-directories |
| `inspect_dataset.py` | An MDS directory | Decoded text printed to stdout |

Each raw dataset is handled by a small adapter file in `dataset/` that provides exactly two functions: `get_files(path) → list[str]` and `get_text(file) → Iterable[list[Record]]`.

---

### `optimus/tokenizer/`

See [tokenizer.md](tokenizer.md) for the full reference. In summary:

| Script | Purpose |
| --- | --- |
| `train_from_scratch.py` | Train a BPE tokenizer from a text corpus |
| `train_from_old.py` | Extend/retrain an existing tokenizer |
| `convert_to_hf.py` | Convert `tokenizer.json` → `AutoTokenizer`-compatible directory |
