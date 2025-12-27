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

## Docker Deployment

Longbow Fletcher provide specialized Docker images for different hardware targets.

### 1. Default (CPU-Only)

The standard `Dockerfile` builds a lightweight, Alpine-based image optimized for CPU execution using OpenBLAS.

```bash
docker build -t longbow-fletcher:latest .
```

### 2. NVIDIA CUDA (MatX)

Optimized for NVIDIA GPUs (A100, H100, T4, etc.). Leverages NVIDIA MatX for kernel fusion and maximum throughput on Linux.

- **File**: `Dockerfile.cuda`
- **Base Image**: `nvidia/cuda:12.2.0-devel-ubuntu22.04`

```bash
docker build -f Dockerfile.cuda -t longbow-fletcher:cuda .
```

### 3. Apple Metal

Optimized for Apple Silicon. While typically run natively on macOS, this Dockerfile can be used for builds or specialized environments.

- **File**: `Dockerfile.metal`

```bash
docker build -f Dockerfile.metal -t longbow-fletcher:metal .
```

## Performance Comparison

...
