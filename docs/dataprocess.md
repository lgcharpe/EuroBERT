# Data Processing

This document covers the data processing pipeline: how raw data files are converted into MosaicML Streaming (MDS) shards ready for training.

## Table of Contents

- [Data Processing](#data-processing)
  - [Table of Contents](#table-of-contents)
  - [Pipeline Overview](#pipeline-overview)
  - [Dataset Adapters](#dataset-adapters)
    - [Built-in Adapters](#built-in-adapters)
    - [Adapter Contract](#adapter-contract)
  - [1. Tokenizing a Dataset](#1-tokenizing-a-dataset)
    - [Usage](#usage)
    - [Parameters](#parameters)
    - [Example](#example)
      - [Fastest Tokenization with `tiktoken`](#fastest-tokenization-with-tiktoken)
  - [2. Packing a Dataset (Optional)](#2-packing-a-dataset-optional)
    - [Usage](#usage-1)
    - [Parameters](#parameters-1)
    - [Example](#example-1)
  - [3. Subsampling a Dataset (Optional)](#3-subsampling-a-dataset-optional)
    - [Usage](#usage-2)
    - [Parameters](#parameters-2)
    - [Example](#example-2)
  - [4. Inspecting a Dataset (Optional)](#4-inspecting-a-dataset-optional)
    - [Usage](#usage-3)
    - [Parameters](#parameters-3)
    - [Example](#example-3)
  - [Data Mix Format](#data-mix-format)
    - [Full Example: Tokenize three datasets and create a mix](#full-example-tokenize-three-datasets-and-create-a-mix)

---

## Pipeline Overview

```
Raw files (parquet / jsonl / …)
        │
        ▼
tokenize_dataset.py     ← uses a dataset/<name>.py adapter
        │  reads text, tokenizes, appends EOS
        ▼
MDS shards  {tokens: int32[], metadata: json}
        │
        ▼ (optional)
pack_dataset.py         ← packs tokens into block_size chunks
        │  produces train/ and val/ sub-trees
        ▼
Packed MDS shards  {tokens: int32[block_size], metadata: json[]}
        │
        ▼ (optional)
subsample_dataset.py    ← splits one MDS tree into sub-indexes
        │  prevents canonical-node timeout for large datasets
        ▼
Sub-indexed MDS ready for training
```

All four scripts are run via `python -m optimus.dataprocess.<script>` and accept parameters as CLI flags (powered by `fire`).

---

## Dataset Adapters

Every raw dataset needs a small adapter file in `optimus/dataprocess/dataset/`. The adapter tells the tokenization pipeline how to:
1. Find all input files for a given directory.
2. Stream records out of each file in batches.

### Built-in Adapters

| Module name | Dataset | Input format |
|---|---|---|
| `agentInstruct` | AgentInstruct | Parquet |
| `ayaInstruct` | Aya Instruct | Parquet |
| `codeBagel` | CodeBagel | Parquet |
| `culturaX` | CulturaX | Parquet |
| `finemath` | FineMath | Parquet |
| `fineweb-edu-2` | FineWeb-Edu-2 | Parquet |
| `fineweb` | FineWeb | Parquet |
| `IndustryCorpus` | IndustryCorpus | Parquet |
| `industryCorpus2` | IndustryCorpus2 | Parquet |
| `languageFiltered` | Language-filtered corpus | Parquet |
| `long_alpaca` | Long Alpaca | Parquet |
| `openPerfectblend` | OpenPerfectBlend | Parquet |
| `orca_agentinstruct` | Orca AgentInstruct | Parquet |
| `parallel` | Parallel text | Parquet |
| `proof-pile-2` | Proof-Pile-2 | Parquet |
| `smolTalk` | SmolTalk | Parquet |
| `the-stack` | The Stack | Parquet |
| `wikipedia` | Wikipedia | Parquet |

### Adapter Contract

Each adapter module must expose exactly two functions:

```python
def get_files(path: str, **kwargs) -> list[str]:
    """Return absolute paths of all data files under `path`."""
    ...

def get_text(file_path: str, **kwargs) -> Iterable[list[dict]]:
    """
    Yield batches of records from `file_path`.
    Each record must be a dict with at least:
      - "text": str          — the raw document text
      - "metadata": dict     — arbitrary JSON-serialisable metadata
    """
    ...
```

**Minimal example** (a plain-text adapter):

```python
# optimus/dataprocess/dataset/mytextfiles.py
from pathlib import Path
from typing import Any, Iterable

def get_files(path: str) -> list[str]:
    return [str(f) for f in Path(path).rglob("*.txt")]

def get_text(file_path: str, batch_size: int = 500) -> Iterable[list[dict[str, Any]]]:
    with open(file_path, encoding="utf-8") as f:
        batch = []
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            batch.append({"text": line, "metadata": {"source": file_path, "line": i}})
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
```

Once saved as `optimus/dataprocess/dataset/mytextfiles.py`, the dataset can be used by passing `--dataset mytextfiles` to `tokenize_dataset.py`. No other changes are needed.

---

## 1. Tokenizing a Dataset

The `tokenize_dataset.py` script tokenizes a dataset using a specified tokenizer and saves the output in an optimized format. The tokenized data can be split into shards based on a size limit and processed in parallel using multiple workers.

### Usage

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir <path> --tokenizer <path_or_name> --dataset <name> [--output_dir <path>] [--size_limit <value>] [--num_workers <num>] [--head <num>] [--read_files_kwargs <json>] [--timeout <seconds>] [--tiktoken]
```

### Parameters

* `input_dir` (*str*): Path to the directory containing the input dataset.
* `tokenizer` (*str*): Name or path of the tokenizer to be used.
* `dataset` (*str*): Name of the dataset to process.
* `output_dir` (*str*, optional): Directory to save the tokenized dataset. Defaults to `./output`.
* `size_limit` (*int | str*, optional): Maximum shard size before creating a new shard. Supports human-readable formats (e.g., `100kb`, `1mb`). Defaults to `64MB` (`1 << 26`).
* `num_workers` (*int | str*, optional): Number of worker processes. Can be set to `max` to use all available CPUs. Defaults to `1`.
* `head` (*int*, optional): Number of batches to process. If not specified, processes the entire dataset.
* `read_files_kwargs` (*dict[str, Any]*, optional): Additional parameters for dataset file reading. Defaults to `None`.
* `timeout` (*int*, optional): Maximum time (in seconds) before termination. Defaults to `None` (no timeout).
* `tiktoken` (*bool*, optional): Whether to use `tiktoken` for faster tokenization. Defaults to `False`.

### Example

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir ./codebagel --tokenizer "meta-llama/Llama-3.1-8B-Instruct" --dataset codeBagel --output_dir ./output --size_limit 100mb --num_workers 4
```

This command tokenizes the `codeBagel` dataset stored in `./codebagel` using the `meta-llama/Llama-3.1-8B-Instruct` tokenizer, saves the output in `./output`, splits shards at 100MB, and runs with 4 workers.

#### Fastest Tokenization with `tiktoken`

For maximum efficiency (up to 2x speedup), use `tiktoken` with a compatible tokenizer and `tokenizer.model` [file](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/original/tokenizer.model):

```bash
python -m optimus.dataprocess.tokenize_dataset --input_dir ./codebagel --tokenizer ./llama_tokenizer.model --dataset codeBagel --output_dir ./tokenized --size_limit 100mb --num_workers 4 --tiktoken
```

---

## 2. Packing a Dataset (Optional)

The `pack_dataset.py` script packs a dataset into blocks (sentences) of a fixed size or uniformly distributed sentence length.

### Usage

```bash
python -m optimus.dataprocess.pack_dataset --local_dir <path> --output_dir <path> [--block_size <int>] [--val_size <int>] [--size_limit <int|str>] [--head <int>] [--num_workers <int>] [--random_size]
```

### Parameters

* `local_dir` (*str*): Local directory path containing the tokenized dataset.
* `output_dir` (*str*): Directory to save the packed dataset.
* `block_size` (*int*, optional): Block size for packing. Defaults to `None`.
* `val_size` (*int*, optional): Validation set size. Defaults to `None`.
* `size_limit` (*int | str*, optional): Size limit for the output files. Defaults to `"64MB"`.
* `head` (*int*, optional): Number of records to process. Defaults to `None`.
* `num_workers` (*int*, optional): Number of workers. Defaults to `1`.
* `random_size` (*bool*, optional): If `True`, selects a random block size between `12` and `block_size`, resulting in a uniformly distributed sentence length. Defaults to `False`.

### Example

```bash
python -m optimus.dataprocess.pack_dataset --local_dir './tokenized' --output_dir './output_pack' --block_size 2048 --random_size
```

---

## 3. Subsampling a Dataset (Optional)

The `subsample_dataset.py` script processes a dataset to split it into subdirectories and merges the indexes. This is particularly useful for optimizing dataset loading, especially in environments like MosaicML, to avoid GPU synchronization timeout errors.

### Usage

```bash
python -m optimus.dataprocess.subsample_dataset --dataset_path <path> --num_shards <int>
```

### Parameters

* `dataset_path` (*str*): The base path where the datasets are located.
* `num_shards` (*int*): The number of shards to split the dataset into.

### Example

```bash
python -m optimus.dataprocess.subsample_dataset --dataset_path "./tokenized" --num_shards 2
```

---

## 4. Inspecting a Dataset (Optional)

The `inspect_dataset.py` script inspects a processed dataset by printing a few samples.

### Usage

```bash
python -m optimus.dataprocess.inspect_dataset --local_dir <path> --tokenizer <path_or_name> [--num_samples <int>]
```

### Parameters

* `local_dir` (*str*): Dataset directory path.
* `tokenizer_name` (*str*): Tokenizer name or path.
* `num_samples` (*int*, optional): Number of samples to print. Defaults to `5`.

### Example

```bash
python -m optimus.dataprocess.inspect_dataset --local_dir './output' --tokenizer "meta-llama/Llama-3.1-8B-Instruct" --num_samples 5
```

---

## Data Mix Format

The trainer consumes datasets through a data mix file, not a single directory. This allows combining multiple tokenized datasets at different sampling weights.

Create a directory (e.g. `./data/mix`) containing a `train.json` file (and optionally `eval.json`). Each file is a JSON array of MosaicML [`Stream`](https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/mixing_data_sources.html) configs:

```json
[
    {
        "local": "/data/tokenized/fineweb",
        "proportion": 0.6
    },
    {
        "local": "/data/tokenized/wikipedia",
        "proportion": 0.2
    },
    {
        "local": "/data/tokenized/codebagel",
        "proportion": 0.2
    }
]
```

Supported fields per stream entry:

| Field | Description |
|---|---|
| `local` | Local path to the MDS shard directory |
| `remote` | Optional remote path (S3/GCS/etc.) for downloading shards |
| `proportion` | Relative sampling weight (normalised across all streams) |
| `repeat` | How many times to repeat the stream (float, default 1.0) |
| `choose` | Fixed number of samples to draw from this stream |
| `download_retry` | Number of download retries |
| `download_timeout` | Download timeout in seconds |

Pass the mix directory to training with `--data_mix_path ./data/mix`.

### Full Example: Tokenize three datasets and create a mix

```bash
# Step 1: tokenize each dataset
python -m optimus.dataprocess.tokenize_dataset \
    --input_dir ./raw/fineweb --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset fineweb --output_dir ./tok/fineweb --num_workers 8 --tiktoken

python -m optimus.dataprocess.tokenize_dataset \
    --input_dir ./raw/wikipedia --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset wikipedia --output_dir ./tok/wikipedia --num_workers 8 --tiktoken

# Step 2: pack each dataset into 2048-token blocks
python -m optimus.dataprocess.pack_dataset \
    --local_dir ./tok/fineweb --output_dir ./packed/fineweb \
    --block_size 2048 --num_workers 8

python -m optimus.dataprocess.pack_dataset \
    --local_dir ./tok/wikipedia --output_dir ./packed/wikipedia \
    --block_size 2048 --num_workers 8

# Step 3: create mix
mkdir -p ./data/mix
cat > ./data/mix/train.json << 'EOF'
[
    {"local": "./packed/fineweb/train",    "proportion": 0.8},
    {"local": "./packed/wikipedia/train",  "proportion": 0.2}
]
EOF

# Step 4: train
python -m optimus.train \
    --model_name eurobert --model_size 210m \
    --tokenizer_path_or_name meta-llama/Meta-Llama-3-8B-Instruct \
    --data_mix_path ./data/mix \
    --batch_size 8 --length 2048 --lr 1e-4 --num_epochs 1 \
    --output_dir ./runs --project_name my_first_run
```