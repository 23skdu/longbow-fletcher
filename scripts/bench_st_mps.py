#!/usr/bin/env python3
"""
Benchmark Sentence Transformers with MPS (Metal) backend.
Generates embeddings for 10K and 20K random sentences.
"""
import time
import torch
from sentence_transformers import SentenceTransformer

# Check MPS availability
if not torch.backends.mps.is_available():
    print("ERROR: MPS not available on this system")
    exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Load model on MPS
model = SentenceTransformer('prajjwal1/bert-tiny', device='mps')

# Generate random sentences (~50 tokens each)
def generate_sentences(n):
    import random
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
             "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
             "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore"]
    return [" ".join(random.choices(words, k=50)) for _ in range(n)]

# Benchmark function
def benchmark(n_vectors):
    sentences = generate_sentences(n_vectors)
    
    # Warmup
    _ = model.encode(sentences[:100])
    torch.mps.synchronize()
    
    # Timed run
    start = time.perf_counter()
    embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    
    throughput = n_vectors / elapsed
    print(f"Sentence Transformers (MPS) - {n_vectors:,} vectors: {elapsed:.2f}s ({throughput:.0f} vec/s)")
    return elapsed, throughput

print("\n--- Sentence Transformers (MPS) Benchmark ---")
benchmark(10_000)
benchmark(20_000)
