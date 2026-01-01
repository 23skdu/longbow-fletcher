import torch
from transformers import AutoModel
from safetensors.torch import save_file
import sys
import os

MODEL_NAME = "prajjwal1/bert-tiny"
OUTPUT_FILE = "bert_tiny.safetensors"

def convert():
    print(f"Loading {MODEL_NAME}...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    tensors = {}
    
    # Helper to add tensor
    def add(name, tensor):
        # Flattening is NOT needed for SafeTensors - it supports shapes!
        # Transpose is handled in loader.go logic via heuristic check.
        # We save standard HF Layout (Out, In).
        
        # We ensure tensors are contiguous and on CPU
        tensors[name] = tensor.detach().cpu().contiguous()
        print(f"Added {name}: {tensors[name].shape}")
        if "word_embeddings" in name:
            print(f"DEBUG {name}[:5]: {tensors[name].flatten()[:5].numpy()}")
        if "attention.self.query.weight" in name:
            # Check Transpose match
            print(f"DEBUG {name} (Transposed)[:5]: {tensors[name].t().flatten()[:5].numpy()}")

    print(f"Extracting weights...")
    
    # 1. Embeddings
    add("embeddings.word_embeddings.weight", model.embeddings.word_embeddings.weight)
    add("embeddings.position_embeddings.weight", model.embeddings.position_embeddings.weight)
    add("embeddings.token_type_embeddings.weight", model.embeddings.token_type_embeddings.weight)
    add("embeddings.LayerNorm.weight", model.embeddings.LayerNorm.weight)
    add("embeddings.LayerNorm.bias", model.embeddings.LayerNorm.bias)
    
    # 2. Encoder Layers
    for i, layer in enumerate(model.encoder.layer):
        prefix = f"encoder.layer.{i}"
        # Attention
        add(f"{prefix}.attention.self.query.weight", layer.attention.self.query.weight)
        add(f"{prefix}.attention.self.query.bias", layer.attention.self.query.bias)
        add(f"{prefix}.attention.self.key.weight", layer.attention.self.key.weight)
        add(f"{prefix}.attention.self.key.bias", layer.attention.self.key.bias)
        add(f"{prefix}.attention.self.value.weight", layer.attention.self.value.weight)
        add(f"{prefix}.attention.self.value.bias", layer.attention.self.value.bias)
        
        add(f"{prefix}.attention.output.dense.weight", layer.attention.output.dense.weight)
        add(f"{prefix}.attention.output.dense.bias", layer.attention.output.dense.bias)
        add(f"{prefix}.attention.output.LayerNorm.weight", layer.attention.output.LayerNorm.weight)
        add(f"{prefix}.attention.output.LayerNorm.bias", layer.attention.output.LayerNorm.bias)
        
        # Intermediate
        add(f"{prefix}.intermediate.dense.weight", layer.intermediate.dense.weight)
        add(f"{prefix}.intermediate.dense.bias", layer.intermediate.dense.bias)
        
        # Output
        add(f"{prefix}.output.dense.weight", layer.output.dense.weight)
        add(f"{prefix}.output.dense.bias", layer.output.dense.bias)
        add(f"{prefix}.output.LayerNorm.weight", layer.output.LayerNorm.weight)
        add(f"{prefix}.output.LayerNorm.bias", layer.output.LayerNorm.bias)
        
    # 3. Pooler
    add("pooler.dense.weight", model.pooler.dense.weight)
    add("pooler.dense.bias", model.pooler.dense.bias)
    
    print(f"Saving to {OUTPUT_FILE}...")
    save_file(tensors, OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    convert()
