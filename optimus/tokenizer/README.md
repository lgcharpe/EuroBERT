# Tokenizer Training

Train a BPE tokenizer from scratch and convert it for use with HuggingFace `AutoTokenizer`.

## Scripts

| Script | Purpose |
|---|---|
| `train_from_scratch.py` | Train a BPE tokenizer on a text corpus |
| `convert_to_hf.py` | Convert the trained tokenizer into an `AutoTokenizer`-compatible directory |

## Quick Start

```bash
# 1. Train a tokenizer
python -m optimus.tokenizer.train_from_scratch \
    --input /path/to/corpus/ \
    --output-dir ./my_tokenizer

# 2. Convert for HuggingFace
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./my_tokenizer/tokenizer.json \
    --output-dir ./my_tokenizer_hf

# 3. Use it
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./my_tokenizer_hf')
print(tok.encode('Hello, world!'))
"
```

## Training (`train_from_scratch.py`)

### Input Formats

The training script accepts `.txt`, `.jsonl`, `.json`, and `.parquet` files. Pass files directly or directories (searched recursively).

```bash
# Single file
python -m optimus.tokenizer.train_from_scratch --input data.jsonl

# Multiple files
python -m optimus.tokenizer.train_from_scratch --input train.txt valid.txt

# Directory (recursively finds .txt, .jsonl, .parquet)
python -m optimus.tokenizer.train_from_scratch --input /data/corpus/

# JSONL/Parquet: specify the text field name
python -m optimus.tokenizer.train_from_scratch --input data.parquet --text-column content

# TXT files: treat each line as a separate document
python -m optimus.tokenizer.train_from_scratch --input corpus.txt --line-by-line
```

### Vocabulary

```bash
# EuroBERT default (128,256 tokens)
python -m optimus.tokenizer.train_from_scratch --input corpus/ --vocab-size 128256

# Smaller tokenizer
python -m optimus.tokenizer.train_from_scratch --input corpus/ --vocab-size 32000

# BERT-sized
python -m optimus.tokenizer.train_from_scratch --input corpus/ --vocab-size 30522

# Require tokens to appear at least 5 times
python -m optimus.tokenizer.train_from_scratch --input corpus/ --min-frequency 5
```

### Special Tokens

Defaults match the Llama-3/EuroBERT convention:

```
<|begin_of_text|>  <|end_of_text|>  <|mask|>  <|parallel_sep|>
<|pad|>            <|unk|>          <|cls|>   <|sep|>
```

Override with:

```bash
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --special-tokens "[CLS]" "[SEP]" "[PAD]" "[UNK]" "[MASK]"
```

### Normalizers

Applied in order. Composable — pass multiple names.

| Name | Description |
|---|---|
| `nfc` | Unicode NFC normalization **(default)** |
| `nfd` | Unicode NFD normalization |
| `nfkc` | Unicode NFKC normalization |
| `nfkd` | Unicode NFKD normalization |
| `lowercase` | Convert to lowercase |
| `strip_accents` | Remove diacritics/accents |
| `bert_normalizer` | BERT-style normalization (clean, lowercase, accents, Chinese chars) |
| `replace` | Regex replacement (requires `--replace-pattern` and `--replace-content`) |

```bash
# Default: NFC only (safe, non-lossy)
python -m optimus.tokenizer.train_from_scratch --input corpus/ --normalizers nfc

# Case-insensitive tokenizer
python -m optimus.tokenizer.train_from_scratch --input corpus/ --normalizers nfc lowercase

# BERT-style
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --normalizers nfc lowercase strip_accents

# Regex replacement (e.g., collapse whitespace)
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --normalizers nfc replace --replace-pattern "\s+" --replace-content " "
```

### Pre-tokenizers

Control how text is split before BPE. Composable — pass multiple names.

| Name | Description |
|---|---|
| `byte_level` | GPT-2/Llama-style byte-level splitting **(default)** — handles all Unicode, never produces UNK |
| `whitespace` | Split on whitespace, isolating punctuation |
| `whitespace_split` | Split on whitespace only (keep punctuation attached) |
| `bert_pre_tokenizer` | BERT-style splitting (whitespace + punctuation) |
| `metaspace` | SentencePiece-style (replace spaces with `▁`) |
| `char_delimiter` | Split on a specific character (requires `--char-delimiter`) |
| `split` | Split on a regex pattern (requires `--split-pattern`) |
| `digits` | Isolate individual digits |
| `punctuation` | Isolate punctuation characters |

```bash
# Default: byte-level (recommended for multilingual)
python -m optimus.tokenizer.train_from_scratch --input corpus/ --pre-tokenizers byte_level

# Whitespace + digits isolation
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --pre-tokenizers whitespace digits

# SentencePiece-style with custom replacement char
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --pre-tokenizers metaspace --metaspace-replacement "▁"

# Add prefix space for byte-level (like RoBERTa)
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --pre-tokenizers byte_level --byte-level-add-prefix-space
```

### Decoder

Auto-inferred from the pre-tokenizer by default. Override with `--decoder`:

| Name | Use with |
|---|---|
| `byte_level` | `byte_level` pre-tokenizer **(default)** |
| `metaspace` | `metaspace` pre-tokenizer |
| `wordpiece` | `bert_pre_tokenizer` |
| `bpe` | Generic BPE fallback |

### Post-processor

Controls how encoded output is structured (e.g., adding `[CLS]`/`[SEP]` tokens).

| Name | Description |
|---|---|
| `byte_level` | ByteLevel offset trimming **(default)** |
| `bert` | Wraps with `[CLS] ... [SEP]` (BERT-style) |
| `roberta` | Wraps with `<s> ... </s>` (RoBERTa-style) |
| `template` | Custom template (requires `--single-template` and/or `--pair-template`) |

```bash
# BERT-style post-processing
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --post-processor bert

# Custom template
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --post-processor template \
    --single-template "<|cls|> \$A <|sep|>" \
    --pair-template "<|cls|> \$A <|sep|> \$B:1 <|sep|>:1"
```

### BPE Trainer Options

```bash
# WordPiece-style continuing subword prefix
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --continuing-subword-prefix "##"

# Limit initial alphabet size
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --limit-alphabet 1000

# Use ByteLevel alphabet as initial alphabet (256 byte values)
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --initial-alphabet-from-pretokenizer

# Cap maximum token length
python -m optimus.tokenizer.train_from_scratch --input corpus/ \
    --max-token-length 16

# Disable progress bar
python -m optimus.tokenizer.train_from_scratch --input corpus/ --no-show-progress
```

### Full Example: EuroBERT-style Tokenizer

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

### Full Example: BERT-style Tokenizer

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

## Conversion (`convert_to_hf.py`)

### Basic Usage

```bash
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./my_tokenizer/tokenizer.json \
    --output-dir ./my_tokenizer_hf
```

### Model Type Presets

The `--model-type` flag adjusts how special tokens are mapped:

| Type | BOS | EOS | UNK | PAD | MASK | CLS | SEP |
|---|---|---|---|---|---|---|---|
| `auto` (default) | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|unk\|>` | `<\|pad\|>` | `<\|mask\|>` | `<\|cls\|>` | `<\|sep\|>` |
| `bert` | `<\|cls\|>` | `<\|sep\|>` | `<\|unk\|>` | `<\|pad\|>` | `<\|mask\|>` | `<\|cls\|>` | `<\|sep\|>` |
| `roberta` | `<\|cls\|>` | `<\|sep\|>` | `<\|unk\|>` | `<\|pad\|>` | `<\|mask\|>` | `<\|cls\|>` | `<\|sep\|>` |
| `gpt2` | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|unk\|>` | `<\|pad\|>` | — | — | — |
| `llama` | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|unk\|>` | `<\|pad\|>` | — | — | — |

```bash
# BERT-style mapping
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./tokenizer.json \
    --output-dir ./hf_bert --model-type bert

# GPT-2 style
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./tokenizer.json \
    --output-dir ./hf_gpt2 --model-type gpt2
```

### Custom Options

```bash
# Custom max sequence length
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./tokenizer.json \
    --output-dir ./hf_tok --model-max-length 512

# Left padding (for decoder models)
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./tokenizer.json \
    --output-dir ./hf_tok --padding-side left

# Override individual special tokens
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./tokenizer.json \
    --output-dir ./hf_tok \
    --bos-token "<s>" --eos-token "</s>"

# Register additional special tokens
python -m optimus.tokenizer.convert_to_hf \
    --tokenizer-path ./tokenizer.json \
    --output-dir ./hf_tok \
    --additional-special-tokens "<|parallel_sep|>" "<|fim_suffix|>"
```

### Output Files

The conversion produces the standard HuggingFace tokenizer files:

```
hf_tokenizer/
├── tokenizer.json           # Full tokenizer definition
├── tokenizer_config.json    # Configuration metadata
└── special_tokens_map.json  # Special token assignments
```

## Recommended Defaults

| Setting | Default | Rationale |
|---|---|---|
| Vocab size | 128,256 | Matches EuroBERT architecture |
| Normalizer | `nfc` | Safe Unicode normalization, non-lossy |
| Pre-tokenizer | `byte_level` | Handles all languages, never produces UNK |
| Decoder | `byte_level` | Matched to pre-tokenizer |
| Post-processor | `byte_level` | Consistent with byte-level pipeline |
| Min frequency | 2 | Filters hapax legomena |
| Model max length | 8,192 | Matches EuroBERT context window |
