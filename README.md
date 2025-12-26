<img width="1024" height="565" alt="image" src="https://github.com/user-attachments/assets/80f12325-820a-44f5-917f-769eb9155176" />

# Longbow Fletcher

**Vector Engine for converting text to vectors for Longbow.**

Fletcher is a high-performance transformer-based embedding engine written in pure Go. It is designed for maximum throughput and efficiency, leveraging hardware acceleration (Metal GPU / CGO BLAS) to achieve state-of-the-art performance on local hardware.

## Key Features

- **Multi-Model Support**: Native support for `bert-tiny` and `nomic-embed-text-v1.5` architectures.
- **Metal GPU Acceleration**: Hand-coded FP16 kernels for Apple Silicon, achieving **~24,000 vec/s** on M3 Pro.
- **High-Performance CPU Backend**: Leverages CGO and hardware-accelerated BLAS (Accelerate/OpenBLAS).
- **Modern Transformer Ops**: Support for **RoPE** (Rotary Positional Embeddings) and **SwiGLU** activation.
- **Pure Go Inference**: Custom transformer implementation—no Python or PyTorch runtime required.
- **Apache Arrow & Flight**: Seamless integration with Longbow via high-performance data transport.
- **Minimal Footprint**: Multi-stage build produces a ~15MB scratch-based Docker image.

## Performance

Fletcher is significantly faster than standard Python/PyTorch implementations on both CPU and GPU.

| Batch Size | Fletcher (CGO/BLAS) | Fletcher (Metal FP16) | PyTorch (CPU) |
|------------|---------------------|-----------------------|---------------|
| 32         | 7,600 vec/s         | **~22,000 vec/s**     | 2,045 vec/s   |
| 64         | 8,276 vec/s         | **~24,000 vec/s**     | 2,199 vec/s   |

*Benchmark on Apple M3 Pro (12 Cores), generating embeddings for `prajjwal1/bert-tiny`.*

## Installation

### Prerequisites

- **macOS**: Xcode Command Line Tools (includes Accelerate framework).
- **Linux**: `libopenblas-dev` (Debian/Ubuntu) or equivalent.

### Build from Source

```bash
# CGO is enabled by default for optimal performance
CGO_ENABLED=1 go build -o bin/fletcher ./cmd/fletcher
```

### Docker

```bash
docker build -t longbow-fletcher .
docker run --rm longbow-fletcher --help
```

## Usage

### CLI Flags

| Flag        | Default            | Description                                           |
|-------------|--------------------|-------------------------------------------------------|
| `--model`   | `bert-tiny`        | Model architecture (`bert-tiny`, `nomic-embed-text`). |
| `--gpu`     | `false`            | Enable Metal GPU acceleration (macOS only).           |
| `--vocab`   | `vocab.txt`        | Path to BERT-style WordPiece vocabulary file.         |
| `--weights` | (none)             | Path to model weights binary.                         |
| `--text`    | (none)             | Text string to embed.                                 |
| `--lorem`   | `0`                | Number of Lorem Ipsum paragraphs to generate.         |
| `--server`  | (none)             | Longbow server address (e.g., `localhost:3000`).      |
| `--dataset` | `fletcher_dataset` | Target dataset name on the Longbow server.            |

### Examples

**Use Nomic-Embed-Text with GPU acceleration:**

```bash
./bin/fletcher --model nomic-embed-text --gpu --vocab vocab.txt --weights nomic.bin --text "Hello world"
```

**Generate and send 100 Lorem Ipsum paragraphs to Longbow:**

```bash
./bin/fletcher --vocab vocab.txt --lorem 100 --server localhost:3000 --dataset test_embeddings
```

## Documentation

- [Fletcher Usage](docs/fletcher.md)
- [Model Support](docs/models.md)
- [GPU Acceleration](docs/gpu.md)
- [Performance Benchmarks](docs/speedtest.md)

## Support

If you find this project useful, please consider [sponsoring me on GitHub](SUPPORT.md).
