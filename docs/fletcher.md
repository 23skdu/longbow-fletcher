# Fletcher Usage

Fletcher is a high-performance CLI for generating vector text embeddings and sending them to a Longbow server.

## Key Features

- **High Performance**: Uses CGO and hardware-accelerated BLAS (Accelerate on macOS, OpenBLAS on Linux) for maximum throughput (~8200 vec/s on M3 Pro).
- **SIMD Optimized**: Critical vector operations leverage SIMD loop unrolling and fast math approximations.
- **Efficient**: Zero-allocation buffer pooling and in-place operations.

## Installation

### Prerequisites

- **macOS**: Xcode Command Line Tools (includes Accelerate framework).
- **Linux**: `libopenblas-dev` (Debian/Ubuntu) or equivalent.

```bash
# Build from source (CGO enabled by default on most systems)
CGO_ENABLED=1 go build -o bin/fletcher ./cmd/fletcher

# Or use Docker
docker build -t longbow-fletcher .
docker run --rm longbow-fletcher --help
```

## CLI Flags

| Flag        | Default            | Description                                      |
|-------------|--------------------|--------------------------------------------------|
| `--vocab`   | `vocab.txt`        | Path to BERT-style WordPiece vocabulary file.    |
| `--weights` | (none)             | Path to model weights binary (optional).         |
| `--text`    | (none)             | Text string to embed.                            |
| `--lorem`   | `0`                | Number of Lorem Ipsum paragraphs to generate.    |
| `--server`  | (none)             | Longbow server address (e.g., `localhost:3000`). |
| `--dataset` | `fletcher_dataset` | Target dataset name on the Longbow server.       |

## Examples

### Generate a single embedding locally

```bash
./bin/fletcher --vocab vocab.txt --text "Hello world"
```

### Generate and send 10 Lorem Ipsum paragraphs to Longbow

```bash
./bin/fletcher --vocab vocab.txt --lorem 10 --server localhost:3000 --dataset my_dataset
```

### Use with Docker

```bash
docker run --rm -v $(pwd)/vocab.txt:/vocab.txt longbow-fletcher \
  --vocab /vocab.txt --lorem 5 --server host.docker.internal:3000
```
