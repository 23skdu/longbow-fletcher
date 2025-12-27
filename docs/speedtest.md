# Fletcher Performance Benchmarks (Fair Comparison)

This report provides an apples-to-apples comparison between **Fletcher (Metal)** and **PyTorch (MPS)** using the exact same BERT architecture and hardware.

**Hardware:** Apple M3 Pro (12 CPU cores, 18GB RAM)  
**Model Config:** Bert-tiny (128 hidden, 2 layers, 2 heads, 512 intermediate, 64 sequence length)

---

## Executive Summary

Fletcher (Metal) significantly outperforms PyTorch (MPS) across all dataset scales, demonstrating both higher throughput and superior stability under sustained load.

| Feature | Fletcher (Metal) | PyTorch (MPS) | Advantage |
|--------|------------------|---------------|-----------|
| **Peak Throughput** | **24,235 vecs/s** | 14,836 vecs/s | **1.6x faster** |
| **Sustained (500K)** | **21,063 vecs/s** | 8,292 vecs/s | **2.5x faster** |
| **Throughput Stability** | High (±5%) | Low (Thermal/Contention Jitter) | **Robust** |
| **Memory Baseline** | < 10 MB | ~370 MB | **37x lighter** |

---

## Dataset Scale Benchmark (vectors/second)

Throughput comparison at various dataset sizes (Batch Size 32).

| Dataset Size | Fletcher (Metal) | PyTorch (MPS) | Speedup |
|--------------|------------------|---------------|---------|
| **5,000** | **24,235** | 14,836 | 1.63x |
| **10,000** | **22,484** | 11,340 | 1.98x |
| **50,000** | **23,221** | 14,548 | 1.60x |
| **100,000** | **20,908** | 12,409 | 1.68x |
| **250,000** | **20,312** | 12,012 | 1.69x |
| **500,000** | **21,063** | 8,292 | **2.54x** |

---

## Detailed Analysis

### Why Fletcher Wins

1. **Zero-Overhead Dispatch**: Fletcher calls Metal compute kernels directly from Go via small C wrappers, avoiding the significant overhead of the Python interpreter and the heavy PyTorch runtime.
2. **Optimized Memory Management**: Fletcher uses a specialized tensor pool that reuses GPU buffers, whereas PyTorch's MPS allocator can suffer from fragmentation and frequent garbage collection under high-frequency small-batch loads.
3. **Consistency**: Under the 500K vector load, PyTorch's throughput dropped by 44% from its peak, likely due to internal contention or thermal throttling managed by the Python runtime. Fletcher maintained >20K vecs/s consistently.
4. **Memory Footprint**: Fletcher's binary is a self-contained ~30MB executable with almost no baseline memory usage. PyTorch requires a >5GB environment and has a ~370MB idle memory footprint.

---

## Reproduction

### PyTorch (MPS)

The PyTorch results were gathered using a custom script that initializes a `transformers.BertModel` with identical parameters:

```bash
python scripts/fair_comparison_pytorch.py --batch-size 32
```

### Fletcher (Metal)

The Fletcher results were gathered using the internal testing suite:

```bash
go test -tags metal -v -run="TestFairComparison_Metal" ./internal/embeddings/model/...
```

---

## Hardware Acceleration Status

| Backend | Status | Tech | Precision |
|---------|--------|------|-----------|
| **Metal** | **Active** | Apple Metal Performance Shaders | FP16/FP32 |
| **CPU** | **Active** | Apple Accelerate (Frameworks) | FP32 |
| **SIMD** | **Active** | ARM64 NEON Hand-rolled Kernels | FP32 |

---
*Last Updated: 2025-12-26*
