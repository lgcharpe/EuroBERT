# Extending the Codebase

This document explains how to add new model architectures, optimizers, learning rate schedulers, dataset adapters, and other components to Optimus.

## Table of Contents

- [Add a Model Architecture](#add-a-model-architecture)
- [Add a Model Size Preset](#add-a-model-size-preset)
- [Add an Optimizer](#add-an-optimizer)
- [Add a Learning Rate Scheduler](#add-a-learning-rate-scheduler)
- [Add a Dataset Adapter](#add-a-dataset-adapter)
- [Add a Configuration Parameter](#add-a-configuration-parameter)
- [Implement Missing Configuration Options](#implement-missing-configuration-options)
- [Add Fused Kernel Support](#add-fused-kernel-support)
- [Use a HuggingFace Model](#use-a-huggingface-model)

---

## Add a Model Architecture

### 1. Create the architecture file

Add a new file under `optimus/trainer/model/encoder/`. Use the existing files as templates:

- [eurobert.py](../optimus/trainer/model/encoder/eurobert.py) — modern encoder (RoPE, RMSNorm, GQA, SwiGLU)
- [bert.py](../optimus/trainer/model/encoder/bert.py) — classical encoder (learned positional embeddings, LayerNorm, GeLU)

Your class must subclass `model.TransformerEncoder` and accept a single `config` dict in `__init__`. It must also expose a `device` property.

**Minimum template:**

```python
# optimus/trainer/model/encoder/mymodel.py
import torch
from optimus.trainer.model import model

mymodel_config = {
    "small": {
        "vocab_size": 128_256,
        "embedding_size": 512,
        "num_head": 8,
        "num_kv_head": 8,
        "num_layer": 6,
        "block_size": 2048,
        "dropout": 0.0,
        "mlp_hidden_dim": 2048,
        "bias": False,
        "rms_norm_eps": 1e-5,
        "attn_impl": "torch",       # "torch" or "flash"
        "rope_base": 10_000,
        "fused_rms_norm": False,
        "fused_rope": False,
        "fused_swiglu": False,
        "fused_cross_entropy": False,
        "tied_weights": False,
    }
}


class MyModel(model.TransformerEncoder):
    def __init__(self, config: dict):
        head_dim = config["embedding_size"] // config["num_head"]
        super().__init__(
            embedding=model.CustomEmbedding(
                config["vocab_size"], config["embedding_size"]
            ),
            blocks=[
                model.Block(
                    attention=model.SelfAttention(   # or FlashSelfAttention
                        embed_dim=config["embedding_size"],
                        head_dim=head_dim,
                        num_heads=config["num_head"],
                        num_kv_heads=config["num_kv_head"],
                        dropout=config["dropout"],
                        block_size=config["block_size"],
                        rope=model.RoPE(
                            dim=head_dim,
                            block_size=config["block_size"],
                            base=config["rope_base"],
                            fused_rope=config["fused_rope"],
                        ),
                        bias=config["bias"],
                        flash=config["attn_impl"] == "flash",
                    ),
                    mlp=model.SwigluMLP(             # or GeluMLP
                        embed_dim=config["embedding_size"],
                        hidden_dim=config["mlp_hidden_dim"],
                        dropout=config["dropout"],
                        bias=config["bias"],
                        fused_swiglu=config["fused_swiglu"],
                    ),
                    attn_norm=model.RMSNorm(         # or nn.LayerNorm
                        config["embedding_size"],
                        eps=config["rms_norm_eps"],
                    ),
                    mlp_norm=model.RMSNorm(
                        config["embedding_size"],
                        eps=config["rms_norm_eps"],
                    ),
                    dropout=config["dropout"],
                )
                for _ in range(config["num_layer"])
            ],
            final_layernorm=model.RMSNorm(
                config["embedding_size"],
                eps=config["rms_norm_eps"],
            ),
            lm_head=torch.nn.Linear(
                config["embedding_size"], config["vocab_size"], bias=config["bias"]
            ),
            fused_cross_entropy=config["fused_cross_entropy"],
        )
        if config["tied_weights"]:
            self.lm_head.weight = self.embedding.weight

    @property
    def device(self):
        return next(self.parameters()).device
```

### 2. Register it in `load.py`

Open [optimus/trainer/model/load.py](../optimus/trainer/model/load.py) and add an import and an `elif` branch in `load_model()`:

```python
# At the top of load.py
from optimus.trainer.model.encoder.mymodel import MyModel, mymodel_config

# Inside load_model(), in the elif chain:
elif config.model.model_name == "mymodel":
    dict_config_model = update_config(
        config.model, mymodel_config[config.model.model_size]
    )
    model = MyModel(dict_config_model)
```

### 3. Use it

```bash
python -m optimus.train \
    --model_name mymodel --model_size small \
    --vocab_size 128256 \
    ...
```

### Available Building Blocks (from `model.py`)

| Class | Description |
|---|---|
| `TransformerEncoder` | Base class: embedding → N×Block → LN → lm_head → loss |
| `Block` | Pre-norm residual block (attn + MLP) |
| `TorchSelfAttention` | Standard multi-head / grouped-query attention |
| `FlashSelfAttention` | Flash Attention 2 variant (requires `flash_attn`) |
| `RoPE` | Rotary positional embedding |
| `RMSNorm` | Root-mean-square normalisation |
| `SwigluMLP` | SwiGLU feed-forward network |
| `GeluMLP` | GELU feed-forward network |
| `CustomEmbedding` | `torch.compile`-friendly token embedding |

If your architecture needs custom components not in `model.py`, define them in your encoder file or in a shared utility module.

---

## Add a Model Size Preset

Each architecture file defines a dict of named size presets (`eurobert_config`, `bert_config`, etc.). To add a new size:

```python
# In optimus/trainer/model/encoder/eurobert.py (example)
eurobert_config = {
    # ... existing sizes ...
    "4b": {
        "vocab_size": 128_256,
        "embedding_size": 3072,
        "num_head": 24,
        "num_kv_head": 8,
        "num_layer": 40,
        "block_size": 2048,
        "dropout": 0.0,
        "mlp_hidden_dim": 8192,
        "bias": False,
        "rms_norm_eps": 1e-5,
        "attn_impl": "torch",
        "rope_base": 10_000,
        "fused_rms_norm": False,
        "fused_rope": False,
        "fused_swiglu": False,
        "fused_cross_entropy": False,
        "tied_weights": False,
    },
}
```

Then use it with `--model_size 4b`. Any key in the size dict can be overridden per-run via CLI flags (e.g. `--embedding_size 2048`).

---

## Add an Optimizer

Optimizer selection is now centralized in [optimus/trainer/optimizer_factory.py](../optimus/trainer/optimizer_factory.py), and `Pretrain.__init__()` calls `build_optimizer(self.model, self.train_config)`.

### 1. Add the optimizer implementation

If the optimizer exists in `torch.optim`, add a branch in `build_optimizer()`.
If it is third-party, import it inside the branch to keep optional dependencies lazy.

Example pattern:

```python
def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if config.optimizer == "AdamW":
        return torch.optim.AdamW(...)
    elif config.optimizer == "MyOptimizer":
        return MyOptimizer(params, ...)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}")
```

### 2. Expose and document the name

`TrainConfig` already exposes `optimizer: str = "AdamW"`. Add your new value to docs and examples.

### 3. Use it from CLI

```bash
python -m optimus.train ... --optimizer SGD
```

---

## Add a Learning Rate Scheduler

### 1. Create the scheduler class (if custom)

If using a PyTorch built-in or third-party class, skip this step. For a custom schedule, subclass `torch.optim.lr_scheduler.LRScheduler`:

```python
# optimus/trainer/script/my_scheduler.py
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class MyScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int, ...):
        ...
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        ...
```

### 2. Register in `get_scheduler()`

In `Pretrain.get_scheduler()` in [pretrain.py](../optimus/trainer/pretrain.py), add an `elif` branch:

```python
elif lr_scheduler == "MyScheduler":
    from optimus.trainer.script.my_scheduler import MyScheduler
    return MyScheduler(
        self.optimizer,
        total_steps=self.train_config.num_epochs * len(self.data.train_dataloader),
        # ... other params from train_config ...
    )
```

### 3. Use it

```bash
python -m optimus.train ... --lr_scheduler MyScheduler
```

---

## Add a Dataset Adapter

A dataset adapter tells `tokenize_dataset.py` how to read a specific raw data format. Each adapter is a single Python file with exactly two functions.

### 1. Create the adapter

Add a file to `optimus/dataprocess/dataset/`. The module name is used as the `--dataset` argument.

```python
# optimus/dataprocess/dataset/my_dataset.py
from pathlib import Path
from typing import Any, Iterable


def get_files(path: str, file_extension: str = "parquet") -> list[str]:
    """Return all data files under `path`."""
    return [str(f) for f in Path(path).rglob(f"*.{file_extension}")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    """
    Yield batches of records.
    Each record must have:
      - "text": str
      - "metadata": any JSON-serialisable value
    """
    import pyarrow.parquet as pq

    f = pq.ParquetFile(file_path)
    for batch in f.iter_batches(batch_size=batch_size, columns=["text", "id"]):
        rows = batch.to_pylist()
        yield [{"text": r["text"], "metadata": {"id": r["id"]}} for r in rows]
    f.close()
```

For JSON-Lines (no extra dependencies needed):

```python
# optimus/dataprocess/dataset/my_jsonl_dataset.py
import json
from pathlib import Path
from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(f) for f in Path(path).rglob("*.jsonl")]


def get_text(file_path: str, batch_size: int = 500) -> Iterable[list[dict[str, Any]]]:
    with open(file_path, encoding="utf-8") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            batch.append({
                "text": record["text"],
                "metadata": {"source": record.get("source", file_path)},
            })
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
```

### 2. Pass extra keyword arguments (optional)

`tokenize_dataset.py` forwards `--read_files_kwargs` as a JSON dict to both `get_files()` and `get_text()`:

```bash
python -m optimus.dataprocess.tokenize_dataset \
    --input_dir ./data --tokenizer ... --dataset my_dataset \
    --read_files_kwargs '{"file_extension": "jsonl", "batch_size": 1000}'
```

The keys are matched by name to your function signatures.

### 3. Use it

```bash
python -m optimus.dataprocess.tokenize_dataset \
    --input_dir ./my_raw_data \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset my_dataset \
    --output_dir ./tokenized/my_dataset \
    --num_workers 4
```

No registration step is needed — the script discovers adapters in the `dataset/` directory automatically.

---

## Add a Configuration Parameter

Configuration parameters live in dataclasses under `optimus/trainer/configuration/`. Every field in any sub-config dataclass is automatically available as a CLI flag for `optimus.train`.

### 1. Choose the right sub-config

| What you're adding | File |
|---|---|
| Training hyperparameter (LR, grad accum, masking, …) | `configuration/train.py` |
| Data loading parameter (batch size, sequence length, …) | `configuration/dataset.py` |
| Architecture parameter (layers, heads, vocab size, …) | `configuration/model.py` |
| Distributed training parameter (sharding, precision, …) | `configuration/distributed.py` |

### 2. Add the field

```python
# In configuration/train.py (example)
@dataclass
class TrainConfig:
    # ... existing fields ...
    my_new_param: float = 0.01          # Add with a default value
    my_optional: Optional[str] = None   # Or optional with None
```

That's it. The `Config.update_config()` method uses `dataclasses.asdict()` to scan all sub-config fields and will automatically pick up the new field from the CLI.

### 3. Use the parameter in code

```python
# In pretrain.py or wherever relevant:
my_value = self.train_config.my_new_param
```

---

## Implement Missing Configuration Options & Fix Known Issues

This section tracks what has been fixed, what still needs implementation, and runtime bugs discovered.

### Implemented Since Previous Audit

1. ✅ `optimizer` is implemented via `build_optimizer()` in `optimus/trainer/optimizer_factory.py` and called from `Pretrain.__init__()` (line 68).
2. ✅ `num_epochs` is now a real training loop bound in `Pretrain.train()` with multi-epoch iteration (line 128).
3. ✅ `train.seed` now applies global RNG seeding in `optimus/train.py` with rank-aware offset (`seed + rank`).
4. ✅ `skip_reload_tensorboard` reset behavior is fixed in `Pretrain.resume()` (line 398, now correctly resets itself).

### Recently Fixed (Audit Round 3)

1. ✅ **Eval context manager crash** — `nullcontext()` instantiated correctly at `optimus/trainer/pretrain.py:249`.
2. ✅ **Eval split loading crash** — `self.eval_streams = None` initialization added at `optimus/trainer/data.py:61` to handle missing `eval.json`.
3. ✅ **CosineAnnealingLR multi-epoch support** — T_max now multiplies by `num_epochs` at `optimus/trainer/pretrain.py:437-438`.

### High-Priority Fixes Needed (Audit Round 9)

1. **HuggingFace + variable-length batches incompatibility**

**Location:** `optimus/trainer/data.py:163-196`, `optimus/trainer/pretrain.py:149-156,264-270`

**Current (BROKEN when `huggingface_id` is set with `var_len=True`):**
```python
return {
    "input_ids" if self.hf_model else "x": inputs,
    "labels": labels,
    "cu_seq_lens": cu_seq_lens,
    "max_seqlen": lengths.max().item(),
}

# Later, HF path forwards full batch:
loss, _ = self.model(**batch)
```

**Issue:** HF models typically accept `input_ids`, `attention_mask`, `labels` etc., but not `cu_seq_lens` / `max_seqlen`. Passing full variable-length batch dict can raise unexpected keyword argument errors.

**Fix options:**
```python
# Option A: disable var_len for HF models
if self.hf_model and self.data_config.var_len:
    raise ValueError("var_len=True is not supported with huggingface_id models.")

# Option B: filter batch keys before HF forward
hf_batch = {k: v for k, v in batch.items() if k in {"input_ids", "labels", "attention_mask", "token_type_ids"}}
loss, _ = self.model(**hf_batch)
```

---

2. **Scheduler steps_per_epoch can be zero**

**Location:** `optimus/trainer/pretrain.py:441,445,452`

**Current (BROKEN for small dataloaders / high grad accumulation):**
```python
steps_per_epoch = len(self.data.train_dataloader) // self.train_config.gradient_accumulation_steps
```

**Issue:** Integer floor division can produce 0, which can break `OneCycleLR`/custom scheduler assumptions.

**Fix:** Clamp to at least 1 and reuse shared variable:
```python
steps_per_epoch = max(
    len(self.data.train_dataloader) // self.train_config.gradient_accumulation_steps,
    1,
)
```

---

### High-Priority Fixes Already Implemented (Audit Round 9)

1. **✅ Global microbatch counter for step_to_skip** (FIXED)
   - Implemented at lines 122, 127, 163
   - Replaces per-epoch calculation
   - Works correctly across epochs and resume

2. **✅ Eval native model batch format** (FIXED)
   - Lines 267-273 extract x and labels correctly
   - Works for both HF and native models

3. **✅ Profiler early exit across loops** (FIXED)
    - Uses early `return` to stop training after profiling
    - No longer continues into later epochs

4. **✅ Eval loss accumulation and averaging** (FIXED)
    - Eval now accumulates `batch_loss` tensor and averages across batches
    - Works for both HF and native branches

5. **✅ `no_sync` guard for non-distributed runs** (FIXED)
    - Uses distributed/method guard before calling `self.model.no_sync()`

6. **✅ Skip-phase barrier guard** (FIXED)
    - `dist.barrier()` in skip completion now only runs when distributed

---

### Medium-Priority Fixes Needed

3. **Enforce data.length in collation/masking**

**Location:** `optimus/trainer/data.py:147-195` (collate functions)

**Current:** Length is only used for throughput accounting (`tokens_per_step`), never to truncate/pad sequences.

**Fix:**
- In `to_torch_collate_fn` and `to_torch_collate_var_len_fn`, check and enforce `self.data_config.length`.
- For `var_len=False`: pad/truncate all sequences to exactly `length`.
- For `var_len=True`: cap total sequence length to not exceed `length * batch_size`.

---

4. **Implement or remove fused_linear_cross_entropy**

**Location:** `optimus/trainer/configuration/model.py:34`, `optimus/trainer/model/load.py`

**Current:** Config field exists but is never consumed.

**Options:**
- **Option A:** Remove from `ModelConfig` if not planned.
- **Option B:** Thread through `load.py` to encoder config and implement fused kernel logic in `TransformerEncoder.forward()`.

---

5. **Use `drop_last=False` for eval dataloader**

**Location:** `optimus/trainer/data.py:141`

**Current:** `drop_last=True` is applied to both train and eval dataloaders.

**Issue:** Validation can silently ignore the final partial batch, biasing eval metrics.

**Fix:** Make `drop_last` conditional by split:
```python
drop_last = not eval
```

or pass a split flag into `__create_dataloader`.

---

### Low-Priority Observations

5. **Remaining steps calculation is not globally accurate**
   - Location: `optimus/trainer/pretrain.py:237`
    - Current formula multiplies by `num_epochs` and still uses current-epoch batch index
   - Minor UX issue, not a functional bug

### Validation & Testing Checklist (Updated)

- [x] Test `step_to_skip` with global microbatch counter and multi-epoch (now working)
- [x] Test `step_to_skip` with checkpoint resume (skip disabled correctly)
- [x] Test eval path with `run_validation=False` (loading gated)
- [x] Test eval path with `run_validation=True` and missing eval.json (gracefully skips)
- [x] Test eval path with both native and HuggingFace models (branching fixed)
- [x] Test CosineAnnealingLR with `num_epochs > 1` (LR correctly decays full training)
- [x] **NEW**: Test gradient accumulation on single-process (no DDP/FSDP) with `gradient_accumulation_steps > 1`
- [x] **NEW**: Test `step_to_skip > 0` on non-distributed run (barrier now guarded)
- [ ] **NEW**: Test HF model with `var_len=True` (should fail fast or filter keys)
- [ ] **NEW**: Test tiny dataset with large `gradient_accumulation_steps` (scheduler should still initialize)
- [x] **NEW**: Test profiling exits correctly when `exit_end_profiling=True` (now fixed with early return)

---

## Add Fused Kernel Support

Optimus has first-class support for [Liger Kernel](https://github.com/linkedin/Liger-Kernel) fused operations. To add a new fused kernel:

### 1. Install the optional dependency

```bash
pip install "git+https://github.com/Nicolas-BZRD/EuroBERT.git[kernel]"
```

### 2. Guard the import

Follow the existing pattern of a try/except at module level:

```python
# In your module file
try:
    from liger_kernel.nn import LigerMyModule
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
```

### 3. Add a config flag and use it

Add a boolean flag to `ModelConfig` (e.g., `fused_my_module: bool = False`) and use it at model construction:

```python
# In your encoder file
if config["fused_my_module"]:
    assert LIGER_AVAILABLE, "Liger kernel is required for fused_my_module."
    my_layer = LigerMyModule(...)
else:
    my_layer = StandardMyModule(...)
```

---

## Use a HuggingFace Model

Any HuggingFace masked LM can be trained with Optimus without writing any model code:

```bash
python -m optimus.train \
    --huggingface_id "bert-base-uncased" \
    --data_mix_path ./data/mix \
    --lr 2e-5 --num_epochs 3 --batch_size 16 --length 512
```

When `huggingface_id` is set:
- The model is loaded with `AutoModelForMaskedLM.from_pretrained(huggingface_id)`.
- The tokenizer is loaded with `AutoTokenizer.from_pretrained(huggingface_id)`.
- All `model_*` architecture config fields are ignored.
- The batch dict uses `"input_ids"` as the input key (HuggingFace convention) rather than `"x"`.

This also works with custom HuggingFace models hosted on the Hub:

```bash
python -m optimus.train \
    --huggingface_id "Nicolas-BZRD/EuroBERT-210m" \
    --data_mix_path ./data/mix \
    --lr 5e-5 --num_epochs 1 --batch_size 8
```
