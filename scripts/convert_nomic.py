#!/usr/bin/env python3
"""Convert nomic-embed-text model from HuggingFace to Fletcher .bin format."""

import torch
from transformers import AutoModel
from safetensors.torch import save_file
import sys
import os
import struct

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
OUTPUT_BIN = "nomic_embed.bin"
OUTPUT_SAFETENSORS = "nomic_embed.safetensors"


def convert_to_safetensors():
    """Download and convert to safetensors format."""
    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tensors = {}

    def add(name, tensor):
        if tensor is not None:
            tensors[name] = tensor.detach().cpu().contiguous()
            print(f"Added {name}: {tensors[name].shape}")

    # Extract all named parameters
    for name, param in model.named_parameters():
        # Skip lm_head (not needed for embeddings)
        if "lm_head" in name:
            continue
        add(name, param)

    # Also try to get embeddings directly
    if hasattr(model, "embeddings"):
        for name, param in model.embeddings.named_parameters():
            add(f"embeddings.{name}", param)

    print(f"\nTotal tensors: {len(tensors)}")
    print(f"\nSaving to {OUTPUT_SAFETENSORS}...")
    save_file(tensors, OUTPUT_SAFETENSORS)
    print("Done!")
    return True


def convert_to_raw_bin():
    """Convert to raw .bin format that Fletcher expects."""
    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    # Get config
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    max_position_embeddings = model.config.max_position_embeddings
    num_hidden_layers = model.config.num_hidden_layers

    print(
        f"Config: hidden_size={hidden_size}, vocab_size={vocab_size}, max_pos={max_position_embeddings}, layers={num_hidden_layers}"
    )

    # Try to identify if it's BERT-like or modern transformer
    has_type_embeddings = hasattr(model, "token_type_embeddings") or hasattr(
        model.embeddings, "token_type_embeddings"
    )

    print(f"Has token type embeddings: {has_type_embeddings}")

    with open(OUTPUT_BIN, "wb") as f:
        # Order must match loader.go: LoadFromRawBinary
        # 1. Embeddings
        print("Writing word_embeddings.weight...")
        w = model.embeddings.word_embeddings.weight.data.cpu().numpy().astype("float32")
        w.tofile(f)

        print("Writing position_embeddings.weight...")
        w = (
            model.embeddings.position_embeddings.weight.data.cpu()
            .numpy()
            .astype("float32")
        )
        w.tofile(f)

        # Token type embeddings (may not exist in all models)
        if has_type_embeddings:
            print("Writing token_type_embeddings.weight...")
            w = (
                model.embeddings.token_type_embeddings.weight.data.cpu()
                .numpy()
                .astype("float32")
            )
        else:
            # Write zeros as placeholder
            print("Writing token_type_embeddings.weight (placeholder)...")
            w = torch.zeros(vocab_size, hidden_size).numpy().astype("float32")
        w.tofile(f)

        # Embedding LayerNorm
        print("Writing embeddings.LayerNorm.weight...")
        w = model.embeddings.LayerNorm.weight.data.cpu().numpy().astype("float32")
        w.tofile(f)
        print("Writing embeddings.LayerNorm.bias...")
        w = model.embeddings.LayerNorm.bias.data.cpu().numpy().astype("float32")
        w.tofile(f)

        # 2. Encoder Layers
        for i in range(num_hidden_layers):
            print(f"Processing layer {i}/{num_hidden_layers}...")
            layer = model.encoder.layer[i]

            # Attention
            print(f"  Layer {i}: attention.self.query.weight...")
            w = layer.attention.self.query.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.self.query.bias...")
            w = layer.attention.self.query.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.self.key.weight...")
            w = layer.attention.self.key.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.self.key.bias...")
            w = layer.attention.self.key.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.self.value.weight...")
            w = layer.attention.self.value.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.self.value.bias...")
            w = layer.attention.self.value.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.output.dense.weight...")
            w = layer.attention.output.dense.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.output.dense.bias...")
            w = layer.attention.output.dense.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: attention.output.LayerNorm.weight...")
            w = (
                layer.attention.output.LayerNorm.weight.data.cpu()
                .numpy()
                .astype("float32")
            )
            w.tofile(f)

            print(f"  Layer {i}: attention.output.LayerNorm.bias...")
            w = (
                layer.attention.output.LayerNorm.bias.data.cpu()
                .numpy()
                .astype("float32")
            )
            w.tofile(f)

            # Intermediate
            print(f"  Layer {i}: intermediate.dense.weight...")
            w = layer.intermediate.dense.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: intermediate.dense.bias...")
            w = layer.intermediate.dense.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

            # Output
            print(f"  Layer {i}: output.dense.weight...")
            w = layer.output.dense.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: output.dense.bias...")
            w = layer.output.dense.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: output.LayerNorm.weight...")
            w = layer.output.LayerNorm.weight.data.cpu().numpy().astype("float32")
            w.tofile(f)

            print(f"  Layer {i}: output.LayerNorm.bias...")
            w = layer.output.LayerNorm.bias.data.cpu().numpy().astype("float32")
            w.tofile(f)

        # 3. Pooler
        print("Writing pooler.dense.weight...")
        w = model.pooler.dense.weight.data.cpu().numpy().astype("float32")
        w.tofile(f)

        print("Writing pooler.dense.bias...")
        w = model.pooler.dense.bias.data.cpu().numpy().astype("float32")
        w.tofile(f)

    file_size = os.path.getsize(OUTPUT_BIN)
    print(f"\nDone! Output: {OUTPUT_BIN} ({file_size / 1024 / 1024:.1f} MB)")
    return True


if __name__ == "__main__":
    convert_to_raw_bin()
