# Architecture Overview

Fletcher is designed as a standalone, high-performance embedding engine that serves as the ingestion layer for the Longbow Vector Database.

## Core Design Principles

1. **Backend Abstraction**: Logic is decoupled from hardware. The `Backend` interface (CPU, Metal, CUDA) handles tensor allocations and operations.
    - **CPU**: Uses OpenBLAS/Accelerate via CGO.
    - **Metal**: Uses custom MSL kernels and MPS Graph on Apple Silicon.
    - **CUDA**: (Planned) Uses NVIDIA MatX/CuBLAS.

2. **Zero-Allocation Path**:
    - **Buffer Pooling**: Tensors are recycled in a `sync.Pool`-like structure to avoid GC pressure.
    - **In-Place Ops**: Activation functions and normalizations modify tensors in place where possible.

3. **Batch-First**: All operations are vectorized for batches. Single inputs are treated as a batch of 1.

## Components

### Embedder

The top-level orchestrator. It manages:

- **Tokenizer**: Splits text into subword IDs.
- **Model**: Executes the Transformer forward pass.
- **Pooling**: Converts token embeddings into a single vector (CLS or Mean).

### Longbow Integration

Fletcher acts as a flight client/server.

- **Ingestion**: Accepts text, generates vectors.
- **Transport**: Sends vectors to Longbow via Apache Arrow Flight (gRPC).
- **Format**: Uses Arrow `FixedSizeList` for zero-copy compatibility.

## Memory Model (Metal)

On Apple Silicon, Fletcher uses:

- **Shared Memory**: Managed buffers accessible by CPU and GPU.
- **FP16 Storage**: Weights and activations stored in Half Precision.
- **FP32 Accumulation**: Critical ops (Softmax, Norm) use FP32 for stability.
