import sys
print("Starting verify_attention.py...", flush=True)

import json
import subprocess
import math

# Lazy import torch/transformers to isolate import issues
print("Importing torch...", flush=True)
import torch
print("Importing transformers...", flush=True)
from transformers import BertModel, BertTokenizer
import numpy as np

print("Imports done.", flush=True)

def main():
    text = "hello world"
    model_name = "prajjwal1/bert-tiny"
    
    # 1. Run Go Script
    print(f"Running Go attention debug for '{text}'...", flush=True)
    try:
        # Debug: Print command
        cmd = ["go", "run", "scripts/debug_attention.go", "-text", text, "-weights", "bert_tiny.bin"]
        # print(f"Command: {cmd}", flush=True)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Go script failed with code {e.returncode}:\nSTDERR:\n{e.stderr}\nSTDOUT:\n{e.stdout}", flush=True)
        sys.exit(1)

    go_dumps = {d["name"]: d for d in json.loads(result.stdout)}

    # 2. PyTorch Reference
    print(f"Running PyTorch reference ({model_name})...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"] 
    
    # Get Embeddings Output
    embed_out = model.embeddings(input_ids)
    
    # Layer 0 Self Attention
    layer0 = model.encoder.layer[0].attention.self
    
    # Projections
    # PyTorch does q = Linear(hidden)
    # shape: (Batch, Seq, Hidden)
    pt_q = layer0.query(embed_out)
    pt_k = layer0.key(embed_out)
    pt_v = layer0.value(embed_out)
    
    # Comparison Helper
    def compare(name, pt_tensor, go_dump, strict=True):
        pt_flat = pt_tensor.detach().numpy().flatten()
        go_data = np.array(go_dump["values"])
        
        sim = 0.0
        if np.linalg.norm(pt_flat) > 0 and np.linalg.norm(go_data) > 0:
            sim = np.dot(pt_flat, go_data) / (np.linalg.norm(pt_flat) * np.linalg.norm(go_data))
        
        diff = np.abs(pt_flat - go_data)
        max_diff = np.max(diff)

        print(f"CHECK {name}: Sim={sim:.6f}, MaxDiff={max_diff:.6f}")
        return sim > 0.99
    
    # Verify Context Layer (Output of Self Attention)
    print("\n--- Context Layer (MHA Logic) ---")
    
    # PyTorch result: (Batch, Seq, Hidden)
    # layer0.self returns (context, scores) if output_attentions=True, but usually just context
    # wait output of 'self' is just context layer.
    pt_context_outputs = layer0(embed_out)
    pt_context = pt_context_outputs[0]
    
    if not compare("context_layer", pt_context, go_dumps["context_layer"]):
        print("FAIL: Context Layer mismatch")
    else:
        print("SUCCESS: Context Layer matches! MHA Bug Fixed.")


if __name__ == "__main__":
    main()
