# GPU Acceleration with Metal

Longbow Fletcher supports high-performance GPU acceleration on Apple Silicon (M1/M2/M3) using the Apple Metal framework.

## Enabling GPU Support

Use the `--gpu` flag to enable Metal acceleration. Fletcher will automatically initialize the Metal backend and offload computational-heavy Transformer layers to the GPU.

```bash
./fletcher --gpu --vocab vocab.txt --text "Hello world"
```

## Features

### 1. FP16 Inference

Fletcher uses **Float16 (Half Precision)** by default on the GPU. This provides:

- **2x Throughput**: Modern Apple Silicon GPUs have specialized hardware for FP16 operations.
- **Lower Memory Footprint**: Halves the VRAM required to store model weights and intermediate activations.
- **High Accuracy**: Fletcher uses Float32 accumulation and synchronization in critical layers (like LayerNorm and Softmax) to maintain high embedding quality.

### 2. Custom Metal Kernels

Fletcher includes hand-coded Metal Shading Language (MSL) kernels for:

- **Matrix Multiplication**: Optimized for Apple Silicon's unified memory architecture.
- **Layer Normalization**: Parallelized with threadgroup reductions.
- **Activation Functions**: Fused kernels for GELU, Tanh, and SwiGLU.
- **RoPE**: Parallel implementation of Rotary Positional Embeddings.

### 3. Buffer Pooling

The Metal backend implements a low-latency buffer pool to minimize expensive GPU memory allocations during inference.

## Performance Comparison

On an Apple M3 Pro, Metal acceleration provides significant speedups over the CPU backend:

| Metric     | CPU (CGO/BLAS) | GPU (Metal FP16) | Speedup   |
|------------|----------------|------------------|-----------|
| Throughput | ~8,200 vec/s   | **~24,000 vec/s** | **~2.9x** |

## Troubleshooting

- **No Metal device found**: Ensure you are running on macOS with Apple Silicon or a Metal-supported Intel Mac.
- **GPU Lockup/VRAM Exhaustion**: If processing extremely large batches or running for prolonged periods, ensure you are using a recent version of Fletcher with built-in buffer synchronization.
