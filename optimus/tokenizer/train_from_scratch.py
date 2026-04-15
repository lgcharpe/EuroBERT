"""Train a BPE tokenizer from scratch using the HuggingFace `tokenizers` library.

Supports extensive configuration of normalization, pre-tokenization, decoding,
and post-processing pipelines via CLI arguments.

Usage:
    python -m optimus.tokenizer.train_from_scratch \
        --input /path/to/corpus \
        --vocab-size 128256 \
        --output-dir ./tokenizer_output

    python -m optimus.tokenizer.train_from_scratch --help
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import (
    NFKC,
    NFKD,
    NFC,
    NFD,
    BertNormalizer,
    Lowercase,
    Replace,
    Sequence as NormalizerSequence,
    StripAccents,
)
from tokenizers.pre_tokenizers import (
    BertPreTokenizer,
    ByteLevel,
    CharDelimiterSplit,
    Digits,
    Metaspace,
    Punctuation,
    Sequence as PreTokenizerSequence,
    Split,
    Whitespace,
    WhitespaceSplit,
)


# ---------------------------------------------------------------------------
# Default special tokens — matches the Llama-3 / EuroBERT convention used in
# optimus/dataprocess/tokenize_dataset.py
# ---------------------------------------------------------------------------
DEFAULT_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|mask|>",
    "<|parallel_sep|>",
    "<|pad|>",
    "<|unk|>",
    "<|cls|>",
    "<|sep|>",
]

NORMALIZER_CHOICES = [
    "nfc",
    "nfd",
    "nfkc",
    "nfkd",
    "lowercase",
    "strip_accents",
    "bert_normalizer",
    "replace",
]

PRE_TOKENIZER_CHOICES = [
    "byte_level",
    "whitespace",
    "whitespace_split",
    "bert_pre_tokenizer",
    "metaspace",
    "char_delimiter",
    "split",
    "digits",
    "punctuation",
]

DECODER_CHOICES = ["byte_level", "metaspace", "wordpiece", "bpe"]

POST_PROCESSOR_CHOICES = ["byte_level", "bert", "roberta", "template"]


# ---------------------------------------------------------------------------
# Corpus iterator
# ---------------------------------------------------------------------------
def iter_corpus(
    inputs: list[str],
    text_column: str = "text",
    line_by_line: bool = False,
) -> Iterator[str]:
    """Yield text strings from files or directories.

    Supported formats: .txt, .jsonl, .json (JSON-Lines), .parquet
    Directories are recursively globbed for supported extensions.
    """
    all_paths: list[str] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.txt", "*.jsonl", "*.json", "*.parquet"):
                all_paths.extend(sorted(glob.glob(str(p / "**" / ext), recursive=True)))
        elif p.is_file():
            all_paths.append(str(p))
        else:
            print(f"Warning: {inp} is not a valid file or directory, skipping.")

    if not all_paths:
        print("Error: no valid input files found.", file=sys.stderr)
        sys.exit(1)

    for fpath in all_paths:
        ext = Path(fpath).suffix.lower()
        if ext == ".txt":
            with open(fpath, encoding="utf-8", errors="replace") as f:
                if line_by_line:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
                else:
                    text = f.read().strip()
                    if text:
                        yield text
        elif ext in (".jsonl", ".json"):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get(text_column, "")
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
        elif ext == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                print(
                    "Error: pyarrow is required for .parquet files. "
                    "Install with: pip install pyarrow",
                    file=sys.stderr,
                )
                sys.exit(1)
            table = pq.read_table(fpath, columns=[text_column])
            for batch in table.to_batches():
                for value in batch.column(text_column):
                    text = value.as_py()
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
        else:
            print(f"Warning: unsupported file extension '{ext}' for {fpath}, skipping.")


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build_normalizer(names: list[str], replace_pattern: str = "", replace_content: str = ""):
    """Build a normalizer pipeline from a list of names."""
    if not names:
        return None

    components = []
    for name in names:
        match name:
            case "nfc":
                components.append(NFC())
            case "nfd":
                components.append(NFD())
            case "nfkc":
                components.append(NFKC())
            case "nfkd":
                components.append(NFKD())
            case "lowercase":
                components.append(Lowercase())
            case "strip_accents":
                components.append(StripAccents())
            case "bert_normalizer":
                components.append(BertNormalizer())
            case "replace":
                if not replace_pattern:
                    print("Warning: --replace-pattern is required for 'replace' normalizer, skipping.")
                    continue
                components.append(Replace(replace_pattern, replace_content))
            case _:
                print(f"Warning: unknown normalizer '{name}', skipping.")

    if len(components) == 0:
        return None
    if len(components) == 1:
        return components[0]
    return NormalizerSequence(components)


def build_pre_tokenizer(
    names: list[str],
    split_pattern: str = "",
    char_delimiter: str = "",
    byte_level_add_prefix_space: bool = False,
    metaspace_replacement: str = "▁",
):
    """Build a pre-tokenizer pipeline from a list of names."""
    if not names:
        return None

    components = []
    for name in names:
        match name:
            case "byte_level":
                components.append(ByteLevel(add_prefix_space=byte_level_add_prefix_space))
            case "whitespace":
                components.append(Whitespace())
            case "whitespace_split":
                components.append(WhitespaceSplit())
            case "bert_pre_tokenizer":
                components.append(BertPreTokenizer())
            case "metaspace":
                components.append(Metaspace(replacement=metaspace_replacement))
            case "char_delimiter":
                if not char_delimiter:
                    print("Warning: --char-delimiter required for 'char_delimiter', skipping.")
                    continue
                components.append(CharDelimiterSplit(char_delimiter))
            case "split":
                if not split_pattern:
                    print("Warning: --split-pattern required for 'split' pre-tokenizer, skipping.")
                    continue
                components.append(Split(pattern=split_pattern, behavior="isolated"))
            case "digits":
                components.append(Digits(individual_digits=True))
            case "punctuation":
                components.append(Punctuation())
            case _:
                print(f"Warning: unknown pre-tokenizer '{name}', skipping.")

    if len(components) == 0:
        return None
    if len(components) == 1:
        return components[0]
    return PreTokenizerSequence(components)


def build_decoder(name: str, metaspace_replacement: str = "▁"):
    """Build a decoder matching the pre-tokenizer."""
    match name:
        case "byte_level":
            return decoders.ByteLevel()
        case "metaspace":
            return decoders.Metaspace(replacement=metaspace_replacement)
        case "wordpiece":
            return decoders.WordPiece()
        case "bpe":
            return decoders.BPEDecoder()
        case _:
            print(f"Warning: unknown decoder '{name}', using ByteLevel.")
            return decoders.ByteLevel()


def infer_decoder(pre_tokenizer_names: list[str]) -> str:
    """Infer a sensible decoder from the pre-tokenizer selection."""
    if "byte_level" in pre_tokenizer_names:
        return "byte_level"
    if "metaspace" in pre_tokenizer_names:
        return "metaspace"
    if "bert_pre_tokenizer" in pre_tokenizer_names:
        return "wordpiece"
    return "byte_level"


def build_post_processor(
    name: str,
    special_tokens: list[str],
    single_template: str = "",
    pair_template: str = "",
):
    """Build a post-processor."""
    match name:
        case "byte_level":
            return processors.ByteLevel(trim_offsets=False)
        case "bert":
            cls_token = "<|cls|>"
            sep_token = "<|sep|>"
            cls_id = special_tokens.index(cls_token) if cls_token in special_tokens else 0
            sep_id = special_tokens.index(sep_token) if sep_token in special_tokens else 1
            return processors.BertProcessing(
                sep=(sep_token, sep_id),
                cls=(cls_token, cls_id),
            )
        case "roberta":
            cls_token = "<|cls|>"
            sep_token = "<|sep|>"
            cls_id = special_tokens.index(cls_token) if cls_token in special_tokens else 0
            sep_id = special_tokens.index(sep_token) if sep_token in special_tokens else 1
            return processors.RobertaProcessing(
                sep=(sep_token, sep_id),
                cls=(cls_token, cls_id),
            )
        case "template":
            st = single_template or "<|cls|> $A <|sep|>"
            pt = pair_template or "<|cls|> $A <|sep|> $B:1 <|sep|>:1"
            token_ids = []
            for tok in special_tokens:
                token_ids.append((tok, special_tokens.index(tok)))
            return processors.TemplateProcessing(
                single=st,
                pair=pt,
                special_tokens=token_ids,
            )
        case _:
            return processors.ByteLevel(trim_offsets=False)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer from scratch using the HuggingFace tokenizers library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Train with defaults (byte-level BPE, NFC normalization, 128k vocab)
  python -m optimus.tokenizer.train_from_scratch \\
      --input /data/corpus/ --output-dir ./my_tokenizer

  # Train a smaller tokenizer with custom normalizers
  python -m optimus.tokenizer.train_from_scratch \\
      --input data.jsonl --vocab-size 32000 \\
      --normalizers nfc lowercase --pre-tokenizers byte_level

  # BERT-style tokenizer with template post-processing
  python -m optimus.tokenizer.train_from_scratch \\
      --input corpus/ --vocab-size 30522 \\
      --normalizers nfc lowercase strip_accents \\
      --pre-tokenizers whitespace punctuation \\
      --post-processor template \\
      --single-template "<|cls|> $A <|sep|>" \\
      --pair-template "<|cls|> $A <|sep|> $B:1 <|sep|>:1"
""",
    )

    # -- Corpus input --
    corpus = parser.add_argument_group("Corpus input")
    corpus.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files or directories (.txt, .jsonl, .parquet). Directories are searched recursively.",
    )
    corpus.add_argument(
        "--text-column",
        default="text",
        help="Column/field name containing text in .jsonl/.parquet files (default: 'text').",
    )
    corpus.add_argument(
        "--line-by-line",
        action="store_true",
        help="For .txt files, treat each line as a separate document (default: whole file = one document).",
    )

    # -- Vocabulary --
    vocab = parser.add_argument_group("Vocabulary")
    vocab.add_argument(
        "--vocab-size",
        type=int,
        default=128_256,
        help="Target vocabulary size (default: 128256, matching EuroBERT).",
    )
    vocab.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token pair to be merged (default: 2).",
    )

    # -- Special tokens --
    sp = parser.add_argument_group("Special tokens")
    sp.add_argument(
        "--special-tokens",
        nargs="+",
        default=DEFAULT_SPECIAL_TOKENS,
        help="List of special tokens (default: Llama-3/EuroBERT-style tokens).",
    )

    # -- Normalizer --
    norm = parser.add_argument_group("Normalizer")
    norm.add_argument(
        "--normalizers",
        nargs="+",
        default=["nfc"],
        choices=NORMALIZER_CHOICES,
        help="Normalizer pipeline components, applied in order (default: nfc).",
    )
    norm.add_argument(
        "--replace-pattern",
        default="",
        help="Regex pattern for 'replace' normalizer.",
    )
    norm.add_argument(
        "--replace-content",
        default="",
        help="Replacement string for 'replace' normalizer.",
    )

    # -- Pre-tokenizer --
    pre = parser.add_argument_group("Pre-tokenizer")
    pre.add_argument(
        "--pre-tokenizers",
        nargs="+",
        default=["byte_level"],
        choices=PRE_TOKENIZER_CHOICES,
        help="Pre-tokenizer pipeline components, applied in order (default: byte_level).",
    )
    pre.add_argument(
        "--split-pattern",
        default="",
        help="Regex pattern for 'split' pre-tokenizer.",
    )
    pre.add_argument(
        "--char-delimiter",
        default="",
        help="Delimiter character for 'char_delimiter' pre-tokenizer.",
    )
    pre.add_argument(
        "--byte-level-add-prefix-space",
        action="store_true",
        help="Add a leading space before the first token in ByteLevel pre-tokenizer.",
    )
    pre.add_argument(
        "--metaspace-replacement",
        default="▁",
        help="Replacement character for Metaspace pre-tokenizer/decoder (default: ▁).",
    )

    # -- Decoder --
    dec = parser.add_argument_group("Decoder")
    dec.add_argument(
        "--decoder",
        default=None,
        choices=DECODER_CHOICES,
        help="Decoder type (default: auto-inferred from pre-tokenizer).",
    )

    # -- Post-processor --
    post = parser.add_argument_group("Post-processor")
    post.add_argument(
        "--post-processor",
        default="byte_level",
        choices=POST_PROCESSOR_CHOICES,
        help="Post-processor type (default: byte_level).",
    )
    post.add_argument(
        "--single-template",
        default="",
        help="Single-sentence template for 'template' post-processor (e.g. '<|cls|> $A <|sep|>').",
    )
    post.add_argument(
        "--pair-template",
        default="",
        help="Sentence-pair template for 'template' post-processor.",
    )

    # -- BPE trainer options --
    tr = parser.add_argument_group("BPE trainer options")
    tr.add_argument(
        "--continuing-subword-prefix",
        default="",
        help="Prefix for continuing subword tokens (e.g. '##' for WordPiece-style).",
    )
    tr.add_argument(
        "--end-of-word-suffix",
        default="",
        help="Suffix appended to end-of-word tokens.",
    )
    tr.add_argument(
        "--limit-alphabet",
        type=int,
        default=None,
        help="Maximum number of characters in the initial alphabet.",
    )
    tr.add_argument(
        "--initial-alphabet-from-pretokenizer",
        action="store_true",
        help="Derive the initial alphabet from the pre-tokenizer (ByteLevel = 256 bytes).",
    )
    tr.add_argument(
        "--show-progress",
        action="store_true",
        default=True,
        help="Show training progress bar (default: True).",
    )
    tr.add_argument(
        "--no-show-progress",
        action="store_true",
        help="Disable training progress bar.",
    )
    tr.add_argument(
        "--max-token-length",
        type=int,
        default=None,
        help="Maximum length (in characters) for a single token.",
    )

    # -- Output --
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output-dir",
        default="./tokenizer_output",
        help="Directory to save the trained tokenizer (default: ./tokenizer_output).",
    )
    out.add_argument(
        "--output-name",
        default="tokenizer.json",
        help="Filename for the tokenizer JSON file (default: tokenizer.json).",
    )

    args = parser.parse_args()

    if args.no_show_progress:
        args.show_progress = False

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("BPE Tokenizer Training")
    print("=" * 60)
    print(f"  Inputs:           {args.input}")
    print(f"  Vocab size:       {args.vocab_size:,}")
    print(f"  Min frequency:    {args.min_frequency}")
    print(f"  Special tokens:   {args.special_tokens}")
    print(f"  Normalizers:      {args.normalizers}")
    print(f"  Pre-tokenizers:   {args.pre_tokenizers}")
    decoder_name = args.decoder or infer_decoder(args.pre_tokenizers)
    print(f"  Decoder:          {decoder_name}")
    print(f"  Post-processor:   {args.post_processor}")
    print(f"  Output:           {args.output_dir}/{args.output_name}")
    print("=" * 60)

    # -- Build tokenizer --
    tokenizer = Tokenizer(models.BPE())

    # Normalizer
    normalizer = build_normalizer(
        args.normalizers,
        replace_pattern=args.replace_pattern,
        replace_content=args.replace_content,
    )
    if normalizer is not None:
        tokenizer.normalizer = normalizer

    # Pre-tokenizer
    pre_tok = build_pre_tokenizer(
        args.pre_tokenizers,
        split_pattern=args.split_pattern,
        char_delimiter=args.char_delimiter,
        byte_level_add_prefix_space=args.byte_level_add_prefix_space,
        metaspace_replacement=args.metaspace_replacement,
    )
    if pre_tok is not None:
        tokenizer.pre_tokenizer = pre_tok

    # Decoder
    tokenizer.decoder = build_decoder(decoder_name, metaspace_replacement=args.metaspace_replacement)

    # Post-processor
    # Note: for bert/roberta/template post-processors the special token IDs are
    # resolved *after* training when the vocab is finalized. We set a placeholder
    # here and reassign after training.
    post_proc_name = args.post_processor

    # -- Build trainer --
    trainer_kwargs: dict = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": args.special_tokens,
        "show_progress": args.show_progress,
    }

    if args.continuing_subword_prefix:
        trainer_kwargs["continuing_subword_prefix"] = args.continuing_subword_prefix
    if args.end_of_word_suffix:
        trainer_kwargs["end_of_word_suffix"] = args.end_of_word_suffix
    if args.limit_alphabet is not None:
        trainer_kwargs["limit_alphabet"] = args.limit_alphabet
    if args.max_token_length is not None:
        trainer_kwargs["max_token_length"] = args.max_token_length

    if args.initial_alphabet_from_pretokenizer and "byte_level" in args.pre_tokenizers:
        trainer_kwargs["initial_alphabet"] = pre_tokenizers.ByteLevel.alphabet()

    trainer = trainers.BpeTrainer(**trainer_kwargs)

    # -- Train --
    print("\nReading corpus and training tokenizer...")
    corpus_iter = iter_corpus(
        args.input,
        text_column=args.text_column,
        line_by_line=args.line_by_line,
    )
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    # -- Set post-processor (after training so vocab IDs are resolved) --
    tokenizer.post_processor = build_post_processor(
        post_proc_name,
        special_tokens=args.special_tokens,
        single_template=args.single_template,
        pair_template=args.pair_template,
    )

    # -- Save --
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    tokenizer.save(output_path)
    print(f"\nTokenizer saved to: {output_path}")

    # -- Summary --
    vocab_size = tokenizer.get_vocab_size()
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Final vocab size: {vocab_size:,}")
    print(f"\nSpecial token IDs:")
    vocab = tokenizer.get_vocab()
    for tok in args.special_tokens:
        tid = vocab.get(tok, None)
        print(f"  {tok:30s} -> {tid}")

    # Sample encoding
    test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Bonjour le monde! 你好世界 🌍",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]
    print(f"\nSample encodings:")
    for s in test_strings:
        encoded = tokenizer.encode(s)
        print(f"  Input:  {s}")
        print(f"  Tokens: {encoded.tokens[:20]}{'...' if len(encoded.tokens) > 20 else ''}")
        print(f"  IDs:    {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Decode: {decoded}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
