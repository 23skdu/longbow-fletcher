# Using Fletcher

## Installation

### Option 1: Docker (Recommended)

The easiest way to run Fletcher is via Docker, ensuring all dependencies are met.

```bash
docker build -t longbow-fletcher .
docker run --rm longbow-fletcher --help
```

### Option 2: Build from Source

For native performance, especially on macOS with Metal support:

```bash
# Clone
git clone https://github.com/23skdu/longbow-fletcher.git
cd longbow-fletcher

# Build (CGO enabled by default for optimizations)
go build -o bin/fletcher ./cmd/fletcher
```

**Prerequisites:**

- **macOS**: Xcode Command Line Tools.
- **Linux**: `libopenblas-dev`.
- **Go**: Version 1.21+.

## Basic Usage (CLI)

Generate embeddings for a single text input:

1. **Download Vocabulary** (BERT-style):

    ```bash
    curl -o vocab.txt https://huggingface.co/bert-base-uncased/raw/main/vocab.txt
    ```

2. **Run Fletcher**:

    ```bash
    ./bin/fletcher --vocab vocab.txt --text "Hello world"
    ```

    *Expected Output:*

    ```text
    ...
    Embedded sequences count=1 dim=128 elapsed=...
    ...
    ```

## Running as a Server

Fletcher can perform as a high-performance HTTP and Arrow Flight server.

**Start the Server:**

```bash
./bin/fletcher -listen :8080 -gpu
```

### Endpoints

- **HTTP POST `/ingest`**: Accepts JSON `{"texts": ["..."]}`.
- **Arrow Flight (Port 9090)**: Use `-flight :9090` to enable. Accepts `DoPut` with `text` column.

### Longbow Integration

Forward embeddings to a persistent Longbow database:

```bash
./bin/fletcher -listen :8080 -server localhost:3000 -dataset my_wiki_data
```

## GPU Acceleration (Metal)

Fletcher natively supports Metal Performance Shaders (MPS) on Apple Silicon (M1/M2/M3).

**Enable GPU:**
Add the `--gpu` flag:

```bash
./bin/fletcher --gpu --vocab vocab.txt --text "Accelerated inference"
```

**Configuration:**

- **FP16 (Default)**: Uses Half Precision for 2x performance and half memory usage.
- **FP32**: Force 32-bit precision with `--precision fp32` (useful for debugging).

**Performance:**
Expect ~2.4x speedup over CPU on M3 Pro chips (e.g., ~24k vs ~10k vectors/sec).

**Troubleshooting:**

- **Crash on Start**: Ensure macOS is updated and avoiding Rosetta.
- **Zero Output**: Verify compatible model config (bert-tiny/nomic).
