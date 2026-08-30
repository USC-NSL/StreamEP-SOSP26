#!/usr/bin/env python3
"""
Download model config + tokenizer from lmsys/gpt-oss-120b-bf16
and reduce layers to 1/4 for testing on 4xA100-40GB.
"""
import json
import os
import shutil
from huggingface_hub import hf_hub_download

MODEL_ID = "lmsys/gpt-oss-120b-bf16"
OUTPUT_DIR = os.path.expanduser("~/gpt-oss-120b-bf16-mini")
LAYER_FRACTION = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

for fname in files:
    try:
        path = hf_hub_download(repo_id=MODEL_ID, filename=fname)
        shutil.copy2(path, os.path.join(OUTPUT_DIR, fname))
        print(f"  Downloaded: {fname}")
    except Exception as e:
        print(f"  Skipped {fname}: {e}")

config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)

original_layers = config["num_hidden_layers"]
new_layers = original_layers // LAYER_FRACTION

print(f"\nReducing layers: {original_layers} -> {new_layers}")

config["num_hidden_layers"] = new_layers
config["layer_types"] = config["layer_types"][:new_layers]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nModel directory ready at: {OUTPUT_DIR}")
print(f"  Layers: {new_layers} (was {original_layers})")
print(f"  Experts: {config.get('num_local_experts', 'N/A')}")
print(f"  Hidden size: {config.get('hidden_size', 'N/A')}")
