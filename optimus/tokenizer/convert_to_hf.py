"""Convert a trained tokenizer (tokenizer.json) into a HuggingFace-compatible
directory loadable via ``AutoTokenizer.from_pretrained()``.

Usage:
    python -m optimus.tokenizer.convert_to_hf \
        --tokenizer-path ./tokenizer_output/tokenizer.json \
        --output-dir ./hf_tokenizer

    python -m optimus.tokenizer.convert_to_hf --help
"""

import argparse
import sys

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Special token mapping presets
# ---------------------------------------------------------------------------
# Maps our default special tokens (from train_from_scratch.py) to the HF
# PreTrainedTokenizerFast keyword arguments.
DEFAULT_TOKEN_MAP = {
    "bos_token": "<|begin_of_text|>",
    "eos_token": "<|end_of_text|>",
    "unk_token": "<|unk|>",
    "pad_token": "<|pad|>",
    "mask_token": "<|mask|>",
    "cls_token": "<|cls|>",
    "sep_token": "<|sep|>",
}

# Per-model-type overrides — adjust which special tokens are mapped.
MODEL_TYPE_OVERRIDES: dict[str, dict[str, str]] = {
    "bert": {
        "bos_token": "<|cls|>",
        "eos_token": "<|sep|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "mask_token": "<|mask|>",
        "cls_token": "<|cls|>",
        "sep_token": "<|sep|>",
    },
    "roberta": {
        "bos_token": "<|cls|>",
        "eos_token": "<|sep|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "mask_token": "<|mask|>",
        "cls_token": "<|cls|>",
        "sep_token": "<|sep|>",
    },
    "gpt2": {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
    },
    "llama": {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a trained tokenizer.json into a HuggingFace-compatible "
            "directory loadable by AutoTokenizer.from_pretrained()."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic conversion (auto-detect model type)
  python -m optimus.tokenizer.convert_to_hf \\
      --tokenizer-path ./tokenizer_output/tokenizer.json \\
      --output-dir ./hf_tokenizer

  # BERT-style conversion with specific max length
  python -m optimus.tokenizer.convert_to_hf \\
      --tokenizer-path ./tokenizer_output/tokenizer.json \\
      --output-dir ./hf_tokenizer \\
      --model-type bert --model-max-length 512

  # Custom special token overrides
  python -m optimus.tokenizer.convert_to_hf \\
      --tokenizer-path ./tokenizer_output/tokenizer.json \\
      --output-dir ./hf_tokenizer \\
      --bos-token "<|begin_of_text|>" \\
      --eos-token "<|end_of_text|>"
""",
    )

    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to the trained tokenizer.json file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the HuggingFace-compatible tokenizer files.",
    )
    parser.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "bert", "roberta", "gpt2", "llama"],
        help=(
            "Model type for special token mapping (default: auto). "
            "'auto' uses the default EuroBERT/Llama-3 mapping."
        ),
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=8192,
        help="Maximum sequence length for the tokenizer (default: 8192, matching EuroBERT).",
    )
    parser.add_argument(
        "--padding-side",
        default="right",
        choices=["left", "right"],
        help="Padding side (default: right).",
    )
    parser.add_argument(
        "--truncation-side",
        default="right",
        choices=["left", "right"],
        help="Truncation side (default: right).",
    )

    # Optional explicit special token overrides
    tok_group = parser.add_argument_group("Special token overrides (optional)")
    tok_group.add_argument("--bos-token", default=None, help="Override BOS token.")
    tok_group.add_argument("--eos-token", default=None, help="Override EOS token.")
    tok_group.add_argument("--unk-token", default=None, help="Override UNK token.")
    tok_group.add_argument("--pad-token", default=None, help="Override PAD token.")
    tok_group.add_argument("--mask-token", default=None, help="Override MASK token.")
    tok_group.add_argument("--cls-token", default=None, help="Override CLS token.")
    tok_group.add_argument("--sep-token", default=None, help="Override SEP token.")

    # Additional tokens
    parser.add_argument(
        "--additional-special-tokens",
        nargs="*",
        default=None,
        help="Additional special tokens to register (e.g. <|parallel_sep|>).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -- Load the tokenizer.json --
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        fast_tokenizer = Tokenizer.from_file(args.tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    vocab = fast_tokenizer.get_vocab()
    print(f"  Vocab size: {fast_tokenizer.get_vocab_size():,}")

    # -- Determine special token mapping --
    if args.model_type == "auto":
        token_map = dict(DEFAULT_TOKEN_MAP)
    else:
        token_map = dict(MODEL_TYPE_OVERRIDES.get(args.model_type, DEFAULT_TOKEN_MAP))

    # Apply CLI overrides
    cli_overrides = {
        "bos_token": args.bos_token,
        "eos_token": args.eos_token,
        "unk_token": args.unk_token,
        "pad_token": args.pad_token,
        "mask_token": args.mask_token,
        "cls_token": args.cls_token,
        "sep_token": args.sep_token,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            token_map[key] = value

    # Only include tokens that exist in the vocabulary
    filtered_map = {}
    for key, tok_str in token_map.items():
        if tok_str in vocab:
            filtered_map[key] = tok_str
        else:
            print(f"  Warning: {key}='{tok_str}' not found in vocab, skipping.")

    # Collect additional special tokens
    additional = []
    if args.additional_special_tokens:
        for tok in args.additional_special_tokens:
            if tok in vocab and tok not in filtered_map.values():
                additional.append(tok)
    else:
        # Auto-detect: include any vocab tokens that look like special tokens
        # but aren't already assigned to a role
        assigned = set(filtered_map.values())
        for tok in vocab:
            if tok.startswith("<|") and tok.endswith("|>") and tok not in assigned:
                additional.append(tok)
        additional.sort(key=lambda t: vocab[t])

    # -- Wrap with PreTrainedTokenizerFast --
    print("\nCreating HuggingFace PreTrainedTokenizerFast wrapper...")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast_tokenizer,
        model_max_length=args.model_max_length,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        additional_special_tokens=additional if additional else None,
        **filtered_map,
    )

    # -- Save --
    hf_tokenizer.save_pretrained(args.output_dir)
    print(f"\nHuggingFace tokenizer saved to: {args.output_dir}")

    # List output files
    import os

    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname:40s} {size:>10,} bytes")

    # -- Verify round-trip --
    print("\nVerification: loading back with AutoTokenizer...")
    from transformers import AutoTokenizer

    loaded = AutoTokenizer.from_pretrained(args.output_dir)
    print(f"  Loaded vocab size: {loaded.vocab_size:,}")
    print(f"  Model max length:  {loaded.model_max_length:,}")

    print(f"\nSpecial tokens:")
    for key in ["bos_token", "eos_token", "unk_token", "pad_token", "mask_token", "cls_token", "sep_token"]:
        tok = getattr(loaded, key, None)
        tid = getattr(loaded, f"{key}_id", None) if tok else None
        if tok:
            print(f"  {key:20s} = {tok!r:30s} (id={tid})")

    if additional:
        print(f"  Additional special tokens ({len(additional)}):")
        for tok in additional[:10]:
            print(f"    {tok} -> {loaded.convert_tokens_to_ids(tok)}")
        if len(additional) > 10:
            print(f"    ... and {len(additional) - 10} more")

    # Round-trip test
    test_strings = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Bonjour le monde! 你好世界 🌍",
    ]
    print(f"\nRound-trip encoding/decoding test:")
    all_ok = True
    for s in test_strings:
        ids = loaded.encode(s, add_special_tokens=False)
        decoded = loaded.decode(ids, skip_special_tokens=True)
        match = "OK" if decoded.strip() == s.strip() else "MISMATCH"
        if match == "MISMATCH":
            all_ok = False
        print(f"  [{match}] {s!r}")
        print(f"         -> ids={ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"         -> decoded={decoded!r}")

    if all_ok:
        print("\nAll round-trip tests passed!")
    else:
        print("\nWarning: some round-trip tests had mismatches (may be expected for byte-level BPE).")

    print(f"\nDone! You can now load the tokenizer with:")
    print(f'  from transformers import AutoTokenizer')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{args.output_dir}")')


if __name__ == "__main__":
    main()
