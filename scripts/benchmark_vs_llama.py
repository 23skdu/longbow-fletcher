import os
import sys
import json
import time
import subprocess
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
FLETCHER_BIN = "./bin/fletcher"
FLETCHER_MODEL = "all-MiniLM-L6-v2"
FLETCHER_WEIGHTS = "model.safetensors"
FLETCHER_PRECISION = "fp16"

LLAMA_URL = "http://localhost:8080/v1/embeddings"

# INPUT_FILE = "coherence_test_input.json"

TEST_SENTENCES = [
  "The quick brown fox jumps over the lazy dog.",
  "A fast brown fox leaps over a sleepy dog.",
  "The weather is nice today.",
  "It is a beautiful day.",
  "Artificial intelligence is transforming the world.",
  "AI is changing everything.",
  "I love pizza.",
  "Pizza is my favorite food."
]

def run_fletcher(sentences):
    print(f"Running Fletcher ({len(sentences)} sequences)...")
    
    # Create temp input file for Fletcher (we need this for CLI)
    tmp_input = "temp_fletcher_input.json"
    with open(tmp_input, 'w') as f:
        # Fletcher CLI expects simple array of strings: ["text1", "text2"]
        json.dump(sentences, f)
        
    cmd = [
        FLETCHER_BIN,
        "-model", FLETCHER_MODEL,
        "-weights", FLETCHER_WEIGHTS,
        "-precision", FLETCHER_PRECISION,
        "-input", tmp_input,
        "-gpu=true"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    end_time = time.time()
    
    if result.returncode != 0:
        print("Fletcher failed:", result.stderr)
        return None, 0
        
    duration = end_time - start_time
    
    # Parse embeddings from stdout
    embeddings = []
    # Fletcher currently logs to stderr, let's capture both
    combined_output = result.stdout + "\n" + result.stderr
    
    # Debug: Print first few lines if verification fails
    # print(combined_output[:500])
    
    for line in combined_output.splitlines():
        try:
            if not line.strip().startswith("{"): continue
            data = json.loads(line)
            if "embedding" in data:
                embeddings.append(data["embedding"])
        except:
            continue
            
    throughput = len(sentences) / duration
    return np.array(embeddings), throughput

def run_llama(sentences):
    print(f"Running llama-server ({len(sentences)} sequences)...")
    
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    try:
        # Llama.cpp embedding endpoint (OpenAI compatible) uses 'input'
        payload = {
            "input": sentences,
            "model": "all-MiniLM-L6-v2.fp16.gguf"
        }
        
        response = requests.post(LLAMA_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Llama output format: { "data": [ { "embedding": [...], "index": 0, ... } ] }
        if "data" not in data:
             print("Llama unexpected response:", data)
             return None, 0
             
        data_list = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in data_list]
        
    except Exception as e:
        print(f"Llama request failed: {e}")
        return None, 0
        
    end_time = time.time()
    duration = end_time - start_time
    throughput = len(sentences) / duration
    
    return np.array(embeddings), throughput

def main():
    sentences = TEST_SENTENCES
    
    # Run Fletcher
    f_embs, f_tps = run_fletcher(sentences)
    print(f"Fletcher Throughput: {f_tps:.2f} seq/sec")
    
    # Run Llama
    l_embs, l_tps = run_llama(sentences)
    print(f"Llama Throughput:    {l_tps:.2f} seq/sec")
    
    if l_tps > 0:
        speedup = f_tps / l_tps
        print(f"Fletcher is {speedup:.2f}x speed of Llama")
    
    if f_embs is not None and l_embs is not None:
         # Compare Coherence
         # Cosine similarity between Fletcher embedding[i] and Llama embedding[i]
         
         if len(f_embs) != len(l_embs):
             print("Mismatch in embedding count")
             return

         sims = []
         print("\nSimilarity Check (Fletcher vs Llama):")
         for i in range(len(f_embs)):
             sim = cosine_similarity([f_embs[i]], [l_embs[i]])[0][0]
             sims.append(sim)
             print(f"Seq {i}: {sim:.4f}")
             
         print(f"\nMean Similarity: {np.mean(sims):.4f}")

if __name__ == "__main__":
    main()
