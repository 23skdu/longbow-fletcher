import time
import sys
from sentence_transformers import SentenceTransformer

# Simple Lorem Ipsum generator to match Fletcher's approx length
LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

def get_texts(n):
    return [LOREM for _ in range(n)]

def benchmark(model, n):
    texts = get_texts(n)
    start = time.time()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, device='cpu')
    end = time.time()
    elapsed = end - start
    rate = n / elapsed
    return rate, elapsed

def main():
    print("Loading model prajjwal1/bert-tiny...")
    # Matches Fletcher's configuration (TinyBERT)
    model = SentenceTransformer('prajjwal1/bert-tiny', device='cpu')
    
    counts = [100, 1000, 10000]
    results = []
    
    print(f"{'Count':<10} {'Time':<10} {'Rate (vec/s)':<15}")
    print("-" * 35)
    
    for n in counts:
        rate, elapsed = benchmark(model, n)
        print(f"{n:<10} {elapsed:.4f}s    {rate:.2f}")
        results.append((n, rate))

if __name__ == "__main__":
    main()
