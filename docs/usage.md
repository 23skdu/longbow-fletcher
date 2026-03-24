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

### Option 3: Model Weights

Fletcher requires model weights in binary format. Use the conversion script:

```bash
# Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install transformers safetensors torch

# Convert BERT-style models (e.g., prajjwal1/bert-tiny)
python scripts/convert_nomic.py  # See script for model-specific conversion
```

Alternatively, download pre-converted weights or use HuggingFace safetensors directly.

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

### CUDA Acceleration (NVIDIA GPUs)

Fletcher supports CUDA on Linux with NVIDIA GPUs.

**Prerequisites:**
- NVIDIA GPU with CUDA toolkit installed
- `nvcc` for compiling CUDA kernels
- cuBLAS library

**Build with CUDA:**
```bash
cd internal/device
nvcc -c cuda_backend.cu -o cuda_backend.o -arch=sm_XX  # XX = your GPU arch (e.g., 86 for RTX 30xx)
nvcc -shared -o libcuda_fletcher.so cuda_backend.o -L/usr/local/cuda/lib64 -lcublas -lcudart -lcuda
cd ../..
CGO_ENABLED=1 go build -tags cuda -o bin/fletcher ./cmd/fletcher
```

**Run with CUDA:**
```bash
# Set library path
export LD_LIBRARY_PATH=/path/to/fletcher/internal/device:$LD_LIBRARY_PATH

# Run with GPU
./bin/fletcher --vocab vocab.txt --weights bert_tiny.bin --text "CUDA inference"
```

**Troubleshooting:**

- **Crash on Start (Mac)**: Ensure macOS is updated and avoiding Rosetta.
- **Zero Output**: Verify compatible model config (bert-tiny/nomic).
- **CUDA Errors**: Ensure `libcuda_fletcher.so` is in `LD_LIBRARY_PATH`.
- **Metal Crash**: Small batch sizes (<16) may trigger MPS bugs; use larger batches for production.
