# Longbow Fletcher

**Vector Engine for converting text to vectors for Longbow.**

Fletcher is a high-performance CLI and library for generating vector text embeddings and sending them to a Longbow server. It is designed for maximum throughput and efficiency, leveraging hardware acceleration where available.

## Key Features

- **High Performance**: Uses CGO and hardware-accelerated BLAS (Accelerate on macOS, OpenBLAS on Linux) for maximum throughput (~8200 vec/s on M3 Pro).
- **Pure Go Embedding Engine**: Custom BERT-style Transformer implementation using `gonum` for linear algebra. No Python required for the core engine.
- **SIMD Optimized**: Critical vector operations leverage SIMD loop unrolling and fast math approximations.
- **Efficient**: Zero-allocation buffer pooling and in-place operations.
- **Apache Arrow Integration**: Formats embeddings as high-performance Arrow RecordBatches.
- **Apache Flight Client**: Sends data to Longbow servers using the Flight RPC protocol.
- **Minimal Docker Image**: Multi-stage build produces a ~15MB image.

## Performance

Fletcher is significantly faster than standard Python/PyTorch implementations on CPU.

| Batch Size | Fletcher (CGO/BLAS) | Sentence Transformers (PyTorch CPU) | Relative Performance |
|------------|---------------------|-------------------------------------|----------------------|
| 32         | **7,600 vec/s**     | 2,045 vec/s                         | **3.7x Faster**      |
| 64         | **8,276 vec/s**     | 2,199 vec/s                         | **3.8x Faster**      |

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

| Flag        | Default            | Description                                      |
|-------------|--------------------|--------------------------------------------------|
| `--vocab`   | `vocab.txt`        | Path to BERT-style WordPiece vocabulary file.    |
| `--weights` | (none)             | Path to model weights binary (optional).         |
| `--text`    | (none)             | Text string to embed.                            |
| `--lorem`   | `0`                | Number of Lorem Ipsum paragraphs to generate.    |
| `--server`  | (none)             | Longbow server address (e.g., `localhost:3000`). |
| `--dataset` | `fletcher_dataset` | Target dataset name on the Longbow server.       |

### Examples

**Generate a single embedding locally:**

```bash
./bin/fletcher --vocab vocab.txt --text "Hello world"
```

**Generate and send 10 Lorem Ipsum paragraphs to Longbow:**

```bash
./bin/fletcher --vocab vocab.txt --lorem 10 --server localhost:3000 --dataset my_dataset
```
