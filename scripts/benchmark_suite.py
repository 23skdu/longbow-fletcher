import os
import json
import time
import sys
import subprocess
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.ipc as ipc

# Configuration
FLETCHER_BIN = "./bin/fletcher"
MODEL_NAME = "prajjwal1/bert-tiny"
INPUT_FILE = "benchmark_input.json"
INPUT_FILE = "fletcher_input.json"

def get_sentences():
    with open(INPUT_FILE, "r") as f:
        return json.load(f)

def run_fletcher(sentences):
    # Input file already exists
    cmd = [FLETCHER_BIN, "-model", "bert-tiny", "-weights", "bert_tiny.safetensors", "-precision", "fp32", "-input", INPUT_FILE, "-gpu=true"]
    
    print(f"Running Fletcher: {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("Fletcher timed out!")
        sys.exit(1)
    end = time.time()
    
    if result.returncode != 0:
        print("Fletcher failed:")
        print(result.stderr.decode())
        sys.exit(1)
        
    elapsed = end - start
    
    # Parse Arrow Output
    # The output might contain logs if not separated correctly, but main.go writes logs to stderr.
    # So stdout should be pure Arrow IPC.
    try:
        reader = ipc.open_stream(result.stdout)
        table = reader.read_all()
        # "embedding" column is a FixedSizeList
        embeddings_col = table.column("embedding")
        # Flatten and reshape? 
        # PyArrow FixedSizeList to numpy
        # Each element is a list of floats.
        embeddings = []
        for batch in table.to_batches():
            d = batch.column("embedding")
            # d is a FixedSizeListArray
            # We can get values as a flat array and reshape
            values = d.values.to_numpy()
            # d.type is fixed_size_list<item: float>[128]
            dim = d.type.list_size
            reshaped = values.reshape(-1, dim)
            embeddings.append(reshaped)
            
        final_embeddings = np.vstack(embeddings)
        return final_embeddings, elapsed
    except Exception as e:
        print(f"Failed to parse Arrow output: {e}")
        # print first 100 bytes of stdout
        print("Stdout preview:", result.stdout[:100])
        sys.exit(1)

def run_transformers_cls(sentences):
    print(f"Loading AutoModel: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Print IDs for debugging
    ids = tokenizer(sentences[0])['input_ids']
    print(f"First Sentence IDs (HF): {ids}")
    
    print("Running Transformers (pooler_output)...")
    start = time.time()
    
    # Process in batch
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return both CLS (raw) and Pooler
    cls_embs = outputs.last_hidden_state[:, 0, :].numpy()
    pooler_embs = outputs.pooler_output.numpy()
    
    end = time.time()
    
    return cls_embs, pooler_embs, end - start

def main():
    print(f"Reading sentences from {INPUT_FILE}...")
    sentences = get_sentences()
    print(f"Loaded {len(sentences)} sentences.")
    
    # 1. Run Fletcher
    print("\n--- Benchmarking Fletcher (Metal) ---")
    fletcher_embs, fletcher_time = run_fletcher(sentences)
    sentences_len = len(sentences)
    fletcher_rate = sentences_len / fletcher_time
    print(f"Fletcher: {sentences_len} vectors in {fletcher_time:.4f}s ({fletcher_rate:.2f} vec/s)")
    
    # 2. Run Transformers
    print("\n--- Benchmarking Transformers (CPU) ---")
    cls_embs, pooler_embs, st_time = run_transformers_cls(sentences)
    st_rate = sentences_len / st_time
    print(f"Transformers: {sentences_len} vectors in {st_time:.4f}s ({st_rate:.2f} vec/s)")
    
    # 3. Correctness
    print("\n--- Verifying Correctness ---")
    
    # Normalize Fletcher
    fletcher_norm = fletcher_embs / np.linalg.norm(fletcher_embs, axis=1, keepdims=True)
    
    # Check Pooler
    st_pooler_norm = pooler_embs / np.linalg.norm(pooler_embs, axis=1, keepdims=True)
    sims_pooler = np.sum(fletcher_norm * st_pooler_norm, axis=1)
    print(f"Pooler Similarity: Mean={np.mean(sims_pooler):.4f}, Min={np.min(sims_pooler):.4f}")

    # Check CLS
    st_cls_norm = cls_embs / np.linalg.norm(cls_embs, axis=1, keepdims=True)
    sims_cls = np.sum(fletcher_norm * st_cls_norm, axis=1)
    print(f"CLS Similarity:    Mean={np.mean(sims_cls):.4f}, Min={np.min(sims_cls):.4f}")
    
    if np.mean(sims_pooler) > 0.99 or np.mean(sims_cls) > 0.99:
        print("SECTION PASSED (Partial or Full Match)")
    else:
        print("SECTION FAILED")

if __name__ == "__main__":
    main()
