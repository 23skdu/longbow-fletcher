# Architecture Overview

Fletcher is a standalone, high-performance embedding engine designed to serve as the ingestion layer for the Longbow Vector Database. It abstracts complex Transformer operations behind a simple CLI and API, optimizing for throughput on modern hardware.

## Core Design Principles

1. **Backend Abstraction**: Logic is decoupled from hardware. The `Backend` interface (CPU, Metal, CUDA) handles tensor allocations and operations.
    - **CPU**: Uses OpenBLAS/Accelerate via CGO for universal compatibility.
    - **Metal**: Uses custom MSL kernels and MPS Graph on Apple Silicon for high-throughput inference.
    - **CUDA**: (Planned) Uses NVIDIA MatX/CuBLAS.

2. **Zero-Allocation Models**:
    - **Buffer Pooling**: Tensors are recycled in `sync.Pool`-like structures to minimize GC pressure.
    - **In-Place Ops**: Activation functions and normalizations modify tensors in place where possible.

3. **Batch-First**: All operations are vectorized for batches. Single inputs are treated as a batch of 1.

## Components

### Embedder Orchestrator

The top-level component that manages the entire pipeline:

1. **Tokenizer**: Splits text into subword IDs using WordPiece.
2. **Model**: Executes the Transformer forward pass.
3. **Pooling**: Converts token embeddings into a single vector (CLS or Mean).

### Longbow Integration

Fletcher acts as a flight client/server:

- **Ingestion**: Accepts text, generates vectors.
- **Transport**: Sends vectors to Longbow via Apache Arrow Flight (gRPC).
- **Format**: Uses Arrow `FixedSizeList` for zero-copy compatibility.

## Model Architecture details

Fletcher uses a custom pure Go implementation of a BERT-style Transformer.

```text
Text Input → Tokenizer → Embeddings (Absolute/RoPE) → Encoder → Pooler → Vector Output
```

### Components Table

| Component | Description |
|-----------|-------------|
| **Tokenizer** | WordPiece tokenizer using BERT-style vocabulary. |
| **Embeddings** | Lookup table mapping token IDs to vectors. Supports **Absolute** (BERT) and **Rotary (RoPE)** (Nomic). |
| **Encoder** | Configurable stack of self-attention + feed-forward layers. Supports **SwiGLU** activation. |
| **Pooler** | Extracts [CLS] token or Mean Pool representation. |

### Weight Initialization

By default (if no weights are provided), Fletcher uses **Xavier/Glorot initialization**:

```text
limit = sqrt(6 / (fan_in + fan_out))
weight ~ Uniform(-limit, limit)
```

## Backend Implementation Details

### Metal (Apple Silicon)

On Apple Silicon, Fletcher leverages the GPU for massive speedups.

- **Shared Memory**: Uses managed buffers accessible by both CPU and GPU to avoid copy overhead.
- **FP16 Storage**: Weights and activations are stored in Half Precision (FP16) to double throughput and halve memory usage.
- **FP32 Accumulation**: Critical operations (Softmax, LayerNorm, Reductions) use FP32 accumulation to prevent overflow and maintain numerical stability.
- **Blit Synchronization**: Uses explicit `MTLBlitCommandEncoder` for safe data transfer between Encoders.

### CPU

- **SIMD**: Uses `gonum` and custom Go assembly loops for vector operations.
- **Parallelism**: Optimizes execution across available cores for batch processing.
