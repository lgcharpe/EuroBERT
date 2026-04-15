# Tokenizer Training

This document covers training a BPE tokenizer from scratch and converting it for use with HuggingFace `AutoTokenizer`. For integrating a trained tokenizer into the data pipeline, see [dataprocess.md](dataprocess.md).

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Train from Scratch](#train-from-scratch)
  - [Input Formats](#input-formats)
  - [Vocabulary](#vocabulary)
  - [Special Tokens](#special-tokens)
  - [Normalizers](#normalizers)
  - [Pre-tokenizers](#pre-tokenizers)
  - [Decoder](#decoder)
  - [Post-processor](#post-processor)
  - [BPE Trainer Options](#bpe-trainer-options)
  - [Full Examples](#full-examples)
- [Train from an Existing Tokenizer](#train-from-an-existing-tokenizer)
- [Convert to HuggingFace Format](#convert-to-huggingface-format)
  - [Model Type Presets](#model-type-presets)
  - [Custom Options](#custom-options)
  - [Output Files](#output-files)
- [Using the Tokenizer for Training](#using-the-tokenizer-for-training)
- [Recommended Defaults](#recommended-defaults)

---

## Overview

The tokenizer pipeline has two steps:

1. **Train** a BPE tokenizer on a raw text corpus with `train_from_scratch.py`.
2. **Convert** the resulting `tokenizer.json` into a HuggingFace-compatible directory with `convert_to_hf.py`, so that `AutoTokenizer.from_pretrained()` can load it — which is what the training pipeline expects.

---

## Installation

`tokenizers` is included in the project dependencies. Verify it is installed:

```bash
python -c "import tokenizers; print(tokenizers.__version__)"
```

---

## Train from Scratch

```bash
python -m optimus.tokenizer.train_from_scratch \
    --input /path/to/corpus/ \
    --output-dir ./my_tokenizer
```

### Input Formats

Pass files or directories. Directories are searched recursively.

| Extension | Handling |
|---|---|
| `.txt` | Whole file = one document; `--line-by-line` for one-document-per-line |
| `.jsonl` / `.json` | JSON-Lines; text extracted from `--text-column` (default: `"text"`) |
| `.parquet` | Column extraction via `--text-column` (default: `"text"`) |

```bash
# Directory (all .txt / .jsonl / .parquet found recursively)
--input /data/corpus/

# Multiple files
--input part1.jsonl part2.jsonl

# JSONL with a non-default column name
--input data.jsonl --text-column content

# TXT line-by-line
--input corpus.txt --line-by-line
```

### Vocabulary

```bash
--vocab-size 128256   # default — matches EuroBERT
--vocab-size 32000    # Llama-2 style
--vocab-size 30522    # BERT style
--min-frequency 2     # ignore pairs seen fewer than 2 times (default)
```

### Special Tokens

Defaults match the Llama-3 / EuroBERT convention used throughout the codebase:

```
<|begin_of_text|>   <|end_of_text|>   <|mask|>   <|parallel_sep|>
<|pad|>             <|unk|>           <|cls|>    <|sep|>
```

Override with `--special-tokens`:

```bash
--special-tokens "[CLS]" "[SEP]" "[PAD]" "[UNK]" "[MASK]"
```

Special tokens are always assigned the lowest IDs (0, 1, 2, …) in the vocabulary, regardless of frequency.

### Normalizers

Applied to the raw input string before tokenization. Composable — list multiple names in order.

| Name | Effect |
|---|---|
| `nfc` | Unicode NFC normalization **(default — safe, non-lossy)** |
| `nfd` | Unicode NFD normalization |
| `nfkc` | Unicode NFKC normalization (compatibility decomposition) |
| `nfkd` | Unicode NFKD normalization |
| `lowercase` | Convert all characters to lowercase |
| `strip_accents` | Remove diacritics (accents) from characters |
| `bert_normalizer` | BERT-style: clean text + lowercase + strip accents + Chinese char handling |
| `replace` | Regex-based replacement (requires `--replace-pattern` and `--replace-content`) |

```bash
# Default
--normalizers nfc

# Case-insensitive
--normalizers nfc lowercase

# BERT-style
--normalizers nfc lowercase strip_accents

# Regex: collapse multiple spaces
--normalizers nfc replace --replace-pattern "\s+" --replace-content " "
```

### Pre-tokenizers

Split the normalised string into "words" before BPE merges. Composable — list multiple names in order.

| Name | Effect |
|---|---|
| `byte_level` | GPT-2/Llama-style — maps every byte to a visible character **(default)** |
| `whitespace` | Split on whitespace; isolate punctuation |
| `whitespace_split` | Split on whitespace only (punctuation stays attached) |
| `bert_pre_tokenizer` | BERT-style: whitespace + punctuation splitting |
| `metaspace` | SentencePiece-style — replaces spaces with `▁` |
| `char_delimiter` | Split on a single character (`--char-delimiter`) |
| `split` | Split on a regex (`--split-pattern`) |
| `digits` | Isolate individual digits |
| `punctuation` | Isolate punctuation characters |

```bash
# Default: byte-level (recommended for multilingual models)
--pre-tokenizers byte_level

# Add space prefix (RoBERTa-style)
--pre-tokenizers byte_level --byte-level-add-prefix-space

# Whitespace + digit isolation
--pre-tokenizers whitespace digits

# SentencePiece style
--pre-tokenizers metaspace --metaspace-replacement "▁"
```

### Decoder

Controls how token IDs are decoded back to text. By default it is **auto-inferred** from the pre-tokenizer:

| Pre-tokenizer | Default decoder |
|---|---|
| `byte_level` | `byte_level` |
| `metaspace` | `metaspace` |
| `bert_pre_tokenizer` | `wordpiece` |
| anything else | `byte_level` |

Override with `--decoder byte_level|metaspace|wordpiece|bpe`.

### Post-processor

Wraps encoded sequences with special tokens.

| Name | Effect |
|---|---|
| `byte_level` | ByteLevel offset trimming **(default)** |
| `bert` | Adds `<\|cls\|>` at start and `<\|sep\|>` at end |
| `roberta` | Same as `bert` but using RoBERTa conventions |
| `template` | Fully configurable via `--single-template` / `--pair-template` |

```bash
# BERT-style wrapping
--post-processor bert

# Custom template
--post-processor template \
  --single-template "<|cls|> \$A <|sep|>" \
  --pair-template "<|cls|> \$A <|sep|> \$B:1 <|sep|>:1"
```

### BPE Trainer Options

```bash
# WordPiece-style continuing subword prefix
--continuing-subword-prefix "##"

# Limit initial alphabet size (reduces memory for huge corpora)
--limit-alphabet 1000

# Seed the initial alphabet from ByteLevel's 256-byte table
--initial-alphabet-from-pretokenizer

# Maximum length of any single token (characters)
--max-token-length 16

# Disable progress bar
--no-show-progress
```

### Full Examples

#### EuroBERT-style tokenizer (multilingual, byte-level BPE)

```bash
python -m optimus.tokenizer.train_from_scratch \
    --input /data/multilingual_corpus/ \
    --vocab-size 128256 \
    --min-frequency 2 \
    --normalizers nfc \
    --pre-tokenizers byte_level \
    --initial-alphabet-from-pretokenizer \
    --output-dir ./eurobert_tokenizer
```

#### BERT-style tokenizer (English, WordPiece-like)

```bash
python -m optimus.tokenizer.train_from_scratch \
    --input /data/english_corpus/ \
    --vocab-size 30522 \
    --min-frequency 2 \
    --special-tokens "[CLS]" "[SEP]" "[PAD]" "[UNK]" "[MASK]" \
    --normalizers nfc lowercase strip_accents \
    --pre-tokenizers bert_pre_tokenizer \
    --decoder wordpiece \
    --post-processor bert \
    --continuing-subword-prefix "##" \
    --output-dir ./bert_tokenizer
```

---

## Train from an Existing Tokenizer

`train_from_old.py` can be used to retrain an existing tokenizer on new data (e.g., to add coverage of a new language or domain) while preserving the original vocabulary structure. Run with `--help` for the full parameter list:

```bash
python -m optimus.tokenizer.train_from_old --help
```

---

## Convert to HuggingFace Format

After training, convert `tokenizer.json` into a directory that `AutoTokenizer.from_pretrained()` can load:

```bash
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./my_tokenizer/tokenizer.json \
    --output-dir ./my_tokenizer_hf
```

The script:
1. Loads the `tokenizers.Tokenizer` from `tokenizer.json`.
2. Wraps it as a `PreTrainedTokenizerFast`.
3. Maps special tokens to their HuggingFace roles (`bos_token`, `eos_token`, `mask_token`, etc.).
4. Saves all files via `save_pretrained()`.
5. Runs a round-trip verification (encode → decode test strings).

### Model Type Presets

`--model-type` controls which vocabulary entries are mapped to which HuggingFace special-token slots:

| `--model-type` | BOS | EOS | UNK | PAD | MASK | CLS | SEP |
|---|---|---|---|---|---|---|---|
| `auto` (default) | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|unk\|>` | `<\|pad\|>` | `<\|mask\|>` | `<\|cls\|>` | `<\|sep\|>` |
| `bert` | `<\|cls\|>` | `<\|sep\|>` | `<\|unk\|>` | `<\|pad\|>` | `<\|mask\|>` | `<\|cls\|>` | `<\|sep\|>` |
| `roberta` | `<\|cls\|>` | `<\|sep\|>` | `<\|unk\|>` | `<\|pad\|>` | `<\|mask\|>` | `<\|cls\|>` | `<\|sep\|>` |
| `gpt2` | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|unk\|>` | `<\|pad\|>` | — | — | — |
| `llama` | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|unk\|>` | `<\|pad\|>` | — | — | — |

### Custom Options

```bash
# Custom max sequence length (default: 8192)
--model-max-length 512

# Left padding (for decoder-style use)
--padding-side left

# Override individual special tokens
--bos-token "<s>" --eos-token "</s>"

# Register extra special tokens explicitly
--additional-special-tokens "<|parallel_sep|>" "<|fim_suffix|>"
```

### Output Files

```
hf_tokenizer/
├── tokenizer.json           # Full tokenizer definition (fast tokenizer)
├── tokenizer_config.json    # HuggingFace tokenizer metadata
└── special_tokens_map.json  # Special token role → string mapping
```

All three files are needed for `AutoTokenizer.from_pretrained()`.

---

## Using the Tokenizer for Training

After conversion, pass the directory path to training:

```bash
python -m optimus.train \
    --tokenizer_path_or_name ./my_tokenizer_hf \
    --mask_token_id <id_of_mask_token> \
    ...
```

The `--mask_token_id` must match the ID of the mask token in your vocabulary. It is printed by `train_from_scratch.py` in the summary output, and also by `convert_to_hf.py` in the special-token table.

---

## Recommended Defaults

| Setting | Default | Rationale |
|---|---|---|
| `--vocab-size` | 128,256 | Matches EuroBERT architecture |
| `--normalizers` | `nfc` | Safe Unicode normalization, no information loss |
| `--pre-tokenizers` | `byte_level` | GPT-2/Llama-style, all-language support, no UNK |
| `--decoder` | auto-inferred | Consistent with pre-tokenizer |
| `--post-processor` | `byte_level` | Correct offset handling for byte-level pipeline |
| `--min-frequency` | 2 | Filters rare token pairs |
| `--model-max-length` | 8,192 | Matches EuroBERT context window |
