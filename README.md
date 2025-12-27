<img width="1024" height="565" alt="image" src="https://github.com/user-attachments/assets/80f12325-820a-44f5-917f-769eb9155176" />

# Longbow Fletcher

**High-Performance Text Embedding Engine for Longbow Vector Database**

Fletcher is a pure Go transformer-based embedding engine designed for maximum throughput on commodity hardware. It converts text into dense vector embeddings using state-of-the-art transformer models, with native hardware acceleration for both Apple Silicon (Metal GPU) and x86 CPUs (BLAS).

## What is Fletcher?

Fletcher is the **vector engine** that feeds [Longbow](https://github.com/23skdu/longbow), a high-performance distributed vector database. While Longbow handles vector storage, indexing (HNSW), and search, Fletcher focuses exclusively on one thing: **converting text to vectors as fast as possible**.

### Key Capabilities

- **Multi-Model Support**: BERT, Nomic-Embed-Text, and custom transformer architectures
- **Metal GPU Acceleration**: Hand-optimized FP16 kernels for Apple Silicon achieving **24,000+ vec/s**
- **CGO BLAS Backend**: Hardware-accelerated CPU inference via Accelerate (macOS) or OpenBLAS (Linux)
- **Modern Transformer Operations**: RoPE (Rotary Positional Embeddings), SwiGLU, LayerNorm
- **Pure Go Implementation**: Zero Python dependencies - pure Go inference pipeline
- **Apache Arrow Integration**: Native Flight protocol for seamless Longbow communication
- **Production-Ready**: Built-in admission control, concurrent request batching, OpenTelemetry support

## Performance

Fletcher significantly outperforms standard PyTorch/SentenceTransformer implementations:

| Metric | Fletcher (Metal) | PyTorch (MPS) | Speedup |
|--------|------------------|---------------|---------|
| **Peak Throughput** | **~24,200 vec/s** | 14,800 vec/s | **1.6x** |
| **Sustained (500K)** | **~21,000 vec/s** | 8,200 vec/s | **2.5x** |
| **Single Latency** | **0.48 ms** | 4.77 ms | **9.9x** |

*Benchmark: Apple M3 Pro (12-core), `prajjwal1/bert-tiny` model, batch size 32.*

For detailed benchmarks including memory usage and CPU performance, see **[Performance Documentation](docs/speedtest.md)**.

## Architecture

Fletcher operates in three modes:

### 1. **Standalone CLI** - Batch Processing

Convert text files or generate embeddings for analysis:

```bash
./fletcher --model nomic-embed-text --gpu --text "Hello world"
```

### 2. **HTTP Server** - RESTful API

Serve embeddings via HTTP with concurrent request batching:

```bash
./fletcher --listen :8080 --model bert-tiny --gpu --max-concurrent 16384
```

### 3. **Flight Server** - Arrow RPC

High-performance gRPC endpoint using Apache Arrow Flight:

```bash
./fletcher --flight :9090 --model nomic-embed-text --gpu
```

## Installation

### Prerequisites

**macOS**: Xcode Command Line Tools

```bash
xcode-select --install
```

**Linux**: OpenBLAS development libraries

```bash
# Debian/Ubuntu
sudo apt-get install libopenblas-dev

# RHEL/CentOS
sudo yum install openblas-devel
```

### Build from Source

```bash
# CGO enabled by default for optimal performance
CGO_ENABLED=1 go build -o bin/fletcher ./cmd/fletcher
```

### Docker

Multi-architecture builds with Metal, CUDA, and CPU backends:

```bash
# CPU-optimized (OpenBLAS)
docker build -f Dockerfile -t fletcher:cpu .

# Metal (Apple Silicon)
docker build -f Dockerfile.metal -t fletcher:metal .

# CUDA (NVIDIA GPUs)
docker build -f Dockerfile.cuda -t fletcher:cuda .
```

## Quick Start

### Basic Usage

```bash
# Embed text with GPU acceleration
./fletcher --model bert-tiny --gpu --vocab vocab.txt --weights bert.bin --text "Machine learning is fascinating"

# Generate 1000 Lorem Ipsum test embeddings
./fletcher --vocab vocab.txt --lorem 1000 --gpu

# Send embeddings to Longbow database
./fletcher --vocab vocab.txt --lorem 100 --server localhost:3000 --dataset my_vectors
```

### Server Mode

```bash
# Start HTTP server
./fletcher --listen :8080 --model nomic-embed-text --gpu --max-vram 4GB

# Start Flight server for Arrow RPC
./fletcher --flight :9090 --vocab vocab.txt --weights nomic.bin
```

### Soak Testing

```bash
# Run sustained load test for 10 minutes
./fletcher --duration 10m --lorem 10000 --gpu
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `bert-tiny` | Model architecture (`bert-tiny`, `nomic-embed-text`) |
| `--gpu` | `false` | Enable Metal GPU acceleration (macOS only) |
| `--vocab` | `vocab.txt` | Path to BERT-style WordPiece vocabulary |
| `--weights` | (required) | Path to model weights binary |
| `--precision` | `fp32` | Compute precision (`fp32`, `fp16`) |
| `--listen` | (none) | HTTP server address (e.g., `:8080`) |
| `--flight` | (none) | Flight server address (e.g., `:9090`) |
| `--server` | (none) | Longbow server endpoint |
| `--dataset` | `fletcher_dataset` | Target dataset name in Longbow |
| `--max-concurrent` | `16384` | Max concurrent embeddings in flight |
| `--max-vram` | `4GB` | VRAM admission control limit |
| `--transport-fmt` | `fp32` | Transport format (`fp32`, `fp16`) |
| `--otel` | `false` | Enable OpenTelemetry tracing |

## Integration with Longbow

Fletcher and Longbow communicate via Apache Arrow Flight for zero-copy data transfer:

```bash
# Terminal 1: Start Longbow server
longbow serve --port 3000

# Terminal 2: Generate and stream embeddings
./fletcher --vocab vocab.txt --lorem 10000 --server localhost:3000 --dataset documents
```

Fletcher outputs Apache Arrow record batches with schema:

```
{
  "text": string,
  "embedding": fixed_size_list<float32>[dim]
}
```

## Documentation

- **[Usage Guide](docs/fletcher.md)** - Detailed CLI and server usage
- **[Model Support](docs/models.md)** - Supported architectures and weights format
- **[GPU Acceleration](docs/gpu.md)** - Metal kernel implementation details
- **[Performance Benchmarks](docs/speedtest.md)** - Comprehensive throughput analysis
- **[API Reference](docs/api.md)** - HTTP and Flight API specifications

## Development

### Running Tests

```bash
# Unit tests
go test -tags metal ./...

# With race detection
go test -tags metal -race ./...

# Coverage
go test -tags metal -coverprofile=coverage.out ./...
```

### Profiling

```bash
# CPU profiling
./fletcher --cpuprofile cpu.pprof --lorem 10000
go tool pprof cpu.pprof

# Memory profiling with pprof server
./fletcher --listen :8080 --gpu
# Visit http://localhost:8080/debug/pprof
```

## Project Structure

```
longbow-fletcher/
├── cmd/fletcher/          # CLI entry point
├── internal/
│   ├── embeddings/        # Embedding engine core
│   ├── device/            # Metal/CPU backend abstraction
│   ├── tokenizer/         # WordPiece tokenizer
│   ├── model/             # Transformer architecture
│   ├── client/            # Arrow Flight client
│   └── server/            # HTTP/Flight servers
├── scripts/               # Benchmark and test scripts
├── helm/                  # Kubernetes deployment
└── docs/                  # Documentation
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Support

If you find this project useful, please consider [sponsoring](SUPPORT.md) to support continued development.

## Related Projects

- **[Longbow](https://github.com/23skdu/longbow)** - Distributed vector database
- **[Longbow-Archer](https://github.com/23skdu/longbow-archer)** - HNSW index implementation
- **[Longbow-Quarrel](https://github.com/23skdu/longbow-quarrel)** - LLM inference engine with Metal backend
