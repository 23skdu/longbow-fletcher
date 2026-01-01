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
import argparse

# Global config
USE_GPU = False

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
    gpu_flag = "-gpu=true" if USE_GPU else "-gpu=false"
    cmd = [FLETCHER_BIN, "-model", "bert-tiny", "-weights", "bert_tiny.safetensors", "-precision", "fp16", "-input", INPUT_FILE, gpu_flag]
    
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
        
        # Parse Stderr for Inference Metrics
        inference_time = elapsed # fallback
        try:
            import re
            stderr_str = result.stderr.decode()
            # Look for: "Embedded sequences" ... elapsed=1.234ms ...
            # "elapsed": "1.2345ms" or similar duration string
            # Log example: {"level":"info","count":4,"elapsed":"13.208µs","dim":128,"tps":302846.76,"time":"..."}
            # zerolog usually outputs JSON or text. ensure we handle text if configured, but main.go uses ConsoleWriter which is text.
            # ConsoleWriter format: <time> INC <caller> > Embedded sequences count=4 elapsed=13.208µs dim=128 tps=302846.76
            
            match = re.search(r'Embedded sequences.*elapsed=(\S+)', stderr_str)
            if match:
                dur_str = match.group(1)
                # Parse duration string to seconds
                if "ms" in dur_str:
                    inference_time = float(dur_str.replace("ms", "")) / 1000.0
                elif "µs" in dur_str or "us" in dur_str:
                    inference_time = float(dur_str.replace("µs", "").replace("us", "")) / 1000000.0
                elif "ns" in dur_str:
                    inference_time = float(dur_str.replace("ns", "")) / 1e9
                elif "s" in dur_str:
                    inference_time = float(dur_str.replace("s", ""))
        except Exception as e:
            print(f"Failed to parse internal metrics: {e}")

        return final_embeddings, elapsed, inference_time
    except Exception as e:
        print(f"Failed to parse Arrow output: {e}")
        # print first 100 bytes of stdout
        print("Stdout preview:", result.stdout[:100])
        sys.exit(1)

def run_transformers_cls(sentences):
    print(f"Loading AutoModel: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    print(f"First Sentence IDs (HF): {tokenizer.encode(sentences[0])}")
    print("Running Transformers (pooler_output)...")
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    pooler_output = outputs.pooler_output.numpy()
    last_hidden_state = outputs.last_hidden_state
    
    # CLS token is at index 0
    cls_output = last_hidden_state[:, 0, :].numpy()
    
    return cls_output, pooler_output, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-tiny")
    parser.add_argument("--weights", default="bert_tiny.safetensors")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    args = parser.parse_args()
    
    global USE_GPU
    USE_GPU = args.gpu
    
    print(f"Reading sentences from {INPUT_FILE}...")
    sentences = get_sentences()
    print(f"Loaded {len(sentences)} sentences.")
    
    # 1. Run Fletcher (Warmup + Performance)
    print(f"\n--- Benchmarking Fletcher (GPU={USE_GPU}) ---")
    
    # Warmup / Check Correctness first
    fletcher_embs, _, _ = run_fletcher(sentences)
    
    # Run loop
    num_runs = 5
    print(f"Running {num_runs} loops for sustained throughput...")
    total_vectors = 0
    total_infer_time = 0.0
    
    for i in range(num_runs):
         _, _, infer_time = run_fletcher(sentences)
         total_vectors += len(sentences)
         total_infer_time += infer_time
    
    avg_infer_time = total_infer_time / num_runs
    sustained_rate = total_vectors / total_infer_time
    print(f"Fletcher (Sustained): {len(sentences)} vectors/batch, {sustained_rate:.2f} vec/s (Avg Infer: {avg_infer_time*1000:.2f}ms)")
    
    # 2. Run Transformers (Reference)
    cls_embs, pooler_embs, _ = run_transformers_cls(sentences)

    if np.mean(fletcher_embs) == 0: # Check for failures
         print("Fletcher output suspicious (zeros).")

    # 3. Correctness
    print("\n--- Verifying Correctness ---")
    
    # Normalize Fletcher
    fletcher_norm = fletcher_embs / (np.linalg.norm(fletcher_embs, axis=1, keepdims=True) + 1e-9)
    
    # Check Pooler
    st_pooler_norm = pooler_embs / (np.linalg.norm(pooler_embs, axis=1, keepdims=True) + 1e-9)
    sims_pooler = np.nan_to_num(np.sum(fletcher_norm * st_pooler_norm, axis=1))
    print(f"Pooler Similarity: Mean={np.mean(sims_pooler):.4f}, Min={np.min(sims_pooler):.4f}")

    # Check CLS
    st_cls_norm = cls_embs / (np.linalg.norm(cls_embs, axis=1, keepdims=True) + 1e-9)
    sims_cls = np.nan_to_num(np.sum(fletcher_norm * st_cls_norm, axis=1))
    print(f"CLS Similarity:    Mean={np.mean(sims_cls):.4f}, Min={np.min(sims_cls):.4f}")
    
    if np.mean(sims_pooler) > 0.99:
        print("SECTION PASSED (Match)")
    else:
        print("SECTION FAILED")

if __name__ == "__main__":
    main()
