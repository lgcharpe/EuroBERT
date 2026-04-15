from __future__ import annotations

import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
import pathlib
import importlib

from typing import Generator



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenizer from an old one.")
    parser.add_argument(
        "--old_tokenizer",
        type=str,
        help="HF path to tokenizer to re-train from (e.g., 'bert-base-uncased' or a local path).",
    )
    parser.add_argument(
        "--new_tokenizer",
        type=str,
        help="Path to save the new tokenizer (e.g., a directory to save tokenizer.json).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a text dataset to use for training the new tokenizer.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        help="Type of dataset to use for training the new tokenizer.",
    )
    parser.add_argument(
        "--hf-dataset-name",
        type=str,
        default=None,
        help="Name of the HuggingFace dataset to use for training the new tokenizer.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30522,
        help="Vocabulary size for the new tokenizer.",
    )
    return parser.parse_args()


def get_training_corpus(args: argparse.Namespace, batch_size: int) -> Generator[list[str], None, None]:
    if args.dataset_path and args.dataset_type:
        # Load dataset using the specified dataset module
        file_dir = pathlib.Path(__file__).resolve().parent.parent / "dataprocess"
        dataset_dir = file_dir / "dataset"
        assert args.dataset_type + ".py" in [f.name for f in dataset_dir.iterdir()], f"{args.dataset_type}.py module not found."
        dataset_module = importlib.import_module(f"optimus.dataprocess.dataset.{args.dataset_type}")
        inputs = dataset_module.get_files(args.dataset_path)
        for inp in inputs:
            for batch in dataset_module.get_text(inp, batch_size=batch_size):
                yield [item["text"] for item in batch]
    elif args.hf_dataset_name:
        # Load dataset from HuggingFace
        raw_datasets = load_dataset(args.hf_dataset_name)
        dataset = raw_datasets["train"] if "train" in raw_datasets else raw_datasets["text"]
        for idx in range(0, len(dataset), batch_size):
            yield dataset[idx:idx + batch_size]["text"]
    else:
        raise ValueError("Either --dataset-path and --dataset-type or --hf-dataset-name must be provided.")


if __name__ == "__main__":
    args = parse_args()

    training_corpus = get_training_corpus(args, batch_size=1000)

    # Load the old tokenizer using HuggingFace's AutoTokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer)

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=args.vocab_size,
    )

    # Save the tokenizer to the new path
    new_tokenizer.save_pretrained(args.new_tokenizer)

    print(f"Tokenizer successfully converted and saved to {args.new_tokenizer}")