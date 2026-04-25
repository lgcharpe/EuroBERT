import json
import os
from collections import OrderedDict
import shutil

import torch
from fire import Fire
from transformers import AutoTokenizer

from typing import Optional

architectures = {
    "base_nor_lumi": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "tie_word_embeddings": False,
        "vocab_size": 51200,
    },
    "base_100k": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "tie_word_embeddings": False,
        "vocab_size": 102400,
    },
    "base_128k": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "tie_word_embeddings": False,
        "vocab_size": 128000,
    },
    "300m": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "tie_word_embeddings": False,
    },
    "500m": {
        "hidden_size": 1152,
        "intermediate_size": 3072,
        "num_hidden_layers": 26,
        "num_attention_heads": 18,
        "num_key_value_heads": 6,
        "tie_word_embeddings": False,
    },
    "2b": {
        "hidden_size": 2304,
        "intermediate_size": 6144,
        "num_hidden_layers": 32,
        "num_attention_heads": 18,
        "num_key_value_heads": 6,
        "tie_word_embeddings": False,
    },
}

base_architecture = {
    "architectures": ["EuroBertModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 4,
    "mask_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1152,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 2048,
    "model_type": "eurobert",
    "num_attention_heads": 18,
    "num_hidden_layers": 26,
    "num_key_value_heads": 6,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.3",
    "use_cache": False,
    "vocab_size": 128256,
    "is_decoder": False,
}

tokenizer_config = {
    "path": "/scratch/project_465002893/nb-embed/EuroBERT/Australis",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "mask_token_id": 2,
}


def remove_prefix_from_state_dict(state_dict, prefixes=["_orig_mod."]):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                break
        new_state_dict[new_key] = value
    return new_state_dict


def convert_state_dict(model_path, config):
    config = {**base_architecture, **config}
    model_dict_weights = torch.load(model_path, map_location=torch.device("cpu"))

    # Remove prefixes from model
    model_dict_weights = remove_prefix_from_state_dict(model_dict_weights)

    new_model = OrderedDict()

    # Compute QKV sizes based on configuration
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    q_size = head_dim * config["num_attention_heads"]
    k_size = head_dim * config["num_key_value_heads"]
    v_size = head_dim * config["num_key_value_heads"]
    qkv_size = q_size + k_size + v_size
    print(f"Q: {q_size}, K: {k_size}, V: {v_size}, QKV: {qkv_size}")

    # Convert model keys to Huggingface format
    prefix = ""
    if not config["tie_word_embeddings"]:
        prefix = "model."

    for k, v in model_dict_weights.items():
        # Rename key properly
        k = (
            k.replace("embedding.weight", "embed_tokens.weight")
            .replace("blocks", "layers")
            .replace(".attn.", ".self_attn.")
            .replace("mlp.fc_1", "mlp.gate_proj")
            .replace("mlp.fc_2", "mlp.up_proj")
            .replace("mlp.proj", "mlp.down_proj")
            .replace("attn_norm", "input_layernorm")
            .replace("mlp_norm", "post_attention_layernorm")
            .replace("self_attn.out_proj", "self_attn.o_proj")
            .replace("final_layernorm", "norm")
        )

        # Handle QKV projections separately
        if ".qkv_proj" in k:
            new_model[prefix + k.replace(".qkv_proj", ".q_proj")] = v[:q_size]
            new_model[prefix + k.replace(".qkv_proj", ".k_proj")] = v[q_size : q_size + k_size]
            new_model[prefix + k.replace(".qkv_proj", ".v_proj")] = v[q_size + k_size :]

        # Handle tied weights in the language model head
        elif "lm_head" in k:
            if config["tie_word_embeddings"]:
                assert (
                    model_dict_weights["lm_head.weight"] == model_dict_weights["embedding.weight"]
                ).all(), "Mismatch in tied weights"
                print("Skipping lm_head as weights are tied")
            else:
                new_model[k] = v

        else:
            new_model[prefix + k] = v

    return new_model, config


def save_hf_model(path, model, config):
    os.makedirs(path, exist_ok=True)
    torch.save(model, f"{path}/pytorch_model.bin")
    with open(f"{path}/config.json", "w") as f:
        json.dump(config, f, indent=2)


def sanitize_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["path"])
    tokenizer.bos_token_id = config["bos_token_id"]
    tokenizer.eos_token_id = config["eos_token_id"]
    tokenizer.mask_token_id = config["mask_token_id"]
    return tokenizer


def save_sanitized_tokenizer(path, tokenizer):
    tokenizer.save_pretrained(path)


def main(original_model_path: str, new_model_path:str, hf_tokenizer_path: Optional[str] = None, hf_sample_path: Optional[str] = None, size: str = "300m"):
    # Convert and save model
    print(f"Converting model to Huggingface format of size {size}.")
    new_model, model_config = convert_state_dict(f"{original_model_path}/model.pt", architectures[size])

    os.makedirs(
        new_model_path,
        exist_ok=True,
    )

    if hf_tokenizer_path is not None:
        shutil.copytree(hf_tokenizer_path, new_model_path, dirs_exist_ok=True)

    if hf_sample_path is not None:
        shutil.copytree(hf_sample_path, new_model_path, dirs_exist_ok=True)

    # Sanitize tokenizer
    # print("Sanitizing tokenizer")
    # new_tokenizer = sanitize_tokenizer(tokenizer_config)

    # Save converted model and sanitized tokenizer
    save_hf_model(new_model_path, new_model, model_config)
    print(f"Saved converted model and sanitized tokenizer to {new_model_path}")


if __name__ == "__main__":
    Fire(main)
