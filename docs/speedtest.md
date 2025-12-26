# Performance Benchmark: Fletcher vs Sentence Transformers

This report compares the performance of **Longbow Fletcher** (Go + Metal/CGO) against **Sentence Transformers** (Python/PyTorch) for embedding generation using the `prajjwal1/bert-tiny` model (L=2, H=128).

## Test Environment

- **Hardware**: Apple M3 Pro (12 Cores)
- **Model**: `prajjwal1/bert-tiny`
- **Task**: Generate embeddings for random Lorem Ipsum paragraphs (~50-100 tokens each).
- **Date**: December 2025

## Results: Throughput (Vectors / Second)

| Batch Size | Fletcher (CGO/BLAS) | Fletcher (Metal FP16) | Sentence Transformers (PyTorch CPU) |
|------------|---------------------|-----------------------|-------------------------------------|
| 32         | 7,600 vec/s         | **~22,000 vec/s**     | 2,045 vec/s                         |
| 64         | 8,276 vec/s         | **~24,000 vec/s**     | 2,199 vec/s                         |

> [!NOTE]
> Fletcher uses CGO to link against hardware-optimized BLAS libraries (Accelerate on macOS, OpenBLAS on Linux) and Metal for GPU acceleration, enabling it to significantly outperform PyTorch on CPU for this workload.

## Key Takeaways

### 1. Speed vs. Portability

Fletcher is significantly faster than standard Python-based inference stacks:

- **Fletcher (Metal/GPU)**: ~24,000 vec/s. Processes **1 million vectors in < 45 seconds**.
- **Fletcher (CGO/CPU)**: ~8,200 vec/s. Processes **1 million vectors in ~2 minutes**.
- **Python (PyTorch CPU)**: ~2,200 vec/s. Processes 1 million vectors in ~8 minutes.

### 2. Startup Time & Footprint

Fletcher wins decisively on operational efficiency:

| Metric           | Fletcher               | Sentence Transformers                  | Winner                      |
|------------------|------------------------|----------------------------------------|-----------------------------|
| **Binary Size**  | ~15 MB (Dynamic)       | > 3 GB (venv + torch)                  | **Fletcher (200x smaller)** |
| **Startup Time** | **28 ms**              | > 3.0 s                                | **Fletcher (100x faster)**  |
| **Dependencies** | libopenblas/Accelerate | Python, PyTorch, Transformers, Drivers | **Fletcher**                |

### 3. Conclusion

- Use **Fletcher (GPU)** for maximum performance on Apple Silicon.
- Use **Fletcher (CPU)** for high-performance extraction on Linux servers without GPUs.
- Use **Sentence Transformers** only for rapid prototyping with experimental models not yet supported by Fletcher.
