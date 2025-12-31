# Performance Benchmarks

**Date**: 2025-12-26
**Hardware**: Apple M3 Pro (12 CPU cores, 18GB RAM)
**Model**: Bert-tiny (128 hidden, 2 layers, 2 heads)

## Comparisons

Fletcher (Metal) vs PyTorch (MPS):

| Feature | Fletcher (Metal) | PyTorch (MPS) | Advantage |
|--------|------------------|---------------|-----------|
| **Peak Throughput** | **24,235 vecs/s** | 14,836 vecs/s | **1.6x faster** |
| **Sustained (500K)** | **21,063 vecs/s** | 8,292 vecs/s | **2.5x faster** |
| **Memory Baseline** | < 10 MB | ~370 MB | **37x lighter** |

## Dataset Scale

| Dataset Size | Fletcher (Metal) | PyTorch (MPS) | Speedup |
|--------------|------------------|---------------|---------|
| **5,000** | **24,235** | 14,836 | 1.63x |
| **500,000** | **21,063** | 8,292 | **2.54x** |

## Key Advantages

1. **Zero-Overhead Dispatch**: Direct CGO calls to Metal kernels.
2. **Optimized Memory**: Buffer pools prevent fragmentation.
3. **Stability**: No thermal throttling or GC pause drops.
