# Getting Started with Fletcher

Fletcher is a high-performance CLI for generating vector text embeddings. This tutorial will guide you through installation and running your first embedding generation.

## Prerequisites

- **macOS**: Xcode Command Line Tools (includes Accelerate framework).
- **Linux**: `libopenblas-dev` (Debian/Ubuntu) or equivalent.
- **Go**: Version 1.21+ (if building from source).

## Installation

### Option 1: Docker (Recommended)

The easiest way to run Fletcher is via Docker.

```bash
docker build -t longbow-fletcher .
docker run --rm longbow-fletcher --help
```

### Option 2: Build from Source

To achieve native performance, especially on macOS with Metal support:

```bash
# Clone the repository
git clone https://github.com/23skdu/longbow-fletcher.git
cd longbow-fletcher

# Build (CGO is enabled by default for BLAS acceleration)
go build -o bin/fletcher ./cmd/fletcher
```

## Your First Embedding

Generate an embedding for the text "Hello world":

1. **Download Vocabulary**: You need a BERT-style vocabulary file.

    ```bash
    curl -o vocab.txt https://huggingface.co/bert-base-uncased/raw/main/vocab.txt
    ```

2. **Run Fletcher**:

    ```bash
    ./bin/fletcher --vocab vocab.txt --text "Hello world"
    ```

    You should see output similar to:

    ```
    ...
    Embedded sequences count=1 dim=128 elapsed=...
    ...
    ```

## Next Steps

- [Enable GPU Acceleration](../how-to/gpu_acceleration.md) to speed up inference.
- [Run Fletcher as a Server](../how-to/run_server.md) for production deployments.
