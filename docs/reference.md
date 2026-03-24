# Reference Guide

## CLI Flags

### General

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-gpu` | `false` | Enable Metal GPU acceleration. |
| `-precision` | `fp16` | Backend precision (`fp16`, `fp32`). |
| `-otel` | `false` | Enable OpenTelemetry tracing. |

### Model Configuration

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-model` | `bert-tiny` | Model type (`bert-tiny`, `nomic-embed-text`). |
| `-vocab` | `vocab.txt` | Path to WordPiece vocabulary. |
| `-weights` | `bert_tiny.bin` | Path to weights file (.bin or .safetensors). |

### Server

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-listen` | `""` | HTTP server address (e.g., `:8080`). |
| `-flight` | `""` | Arrow Flight server address (e.g., `:9090`). |
| `-server` | `""` | Remote Longbow Flight address for forwarding. |
| `-max-concurrent` | `16384` | Maximum concurrent embeddings in flight. |
| `-max-vram` | `4GB` | VRAM admission control limit. |
| `-transport-fmt` | `fp32` | Transport format (`fp32`, `fp16`). |

### Performance

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-duration` | `""` | Run soak test for specified duration (e.g., `10s`, `20m`). |
| `-lorem` | `0` | Generate N lines of lorem ipsum for testing. |
| `-input` | `""` | Path to input file (JSON array of strings). |

---

## Supported Models

| Model | Dimensions | Layers | Heads | Max Seq | Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT-Tiny** | 128 | 2 | 2 | 512 | Absolute Pos |
| **Nomic-v1.5** | 768 | 12 | 12 | 8192 | **RoPE**, **SwiGLU** |

### Special Features

- **RoPE**: Rotary Positional Embeddings allow for long contexts (8k+).
- **SwiGLU**: Activation function used in Nomic, accelerated by custom Metal kernels.

### Weight Formats

Fletcher supports two weight formats:

| Format | Extension | Description |
| :--- | :--- | :--- |
| **Binary** | `.bin` | Raw float32 weights in Fletcher's internal order (see `weights/loader.go`). |
| **SafeTensors** | `.safetensors` | HuggingFace format with automatic transpose handling. |

**Note**: Nomic-embed-text is NOT a BERT architecture and requires model architecture updates to support.

---

## Metrics (Prometheus)

Exposed at `/metrics` when running in server mode.

### GPU / Throughput

| Metric | Type | Description |
| :--- | :--- | :--- |
| `fletcher_gpu_throughput` | Gauge | Sequences processed per second. |
| `fletcher_vectors_processed_total` | Counter | Total vectors generated. |
| `fletcher_request_duration_seconds` | Histogram | End-to-end request latency. |

### Memory

| Metric | Type | Description |
| :--- | :--- | :--- |
| `fletcher_vram_allocated_bytes` | Gauge | Current VRAM usage (Metal) or Heap (CPU). |
| `fletcher_metal_pool_hits_total` | Counter | Buffer pool hits (avoided allocations). |

### Errors

| Metric | Type | Description |
| :--- | :--- | :--- |
| `fletcher_output_invalid_total` | Counter | Total output batches containing NaNs. |
| `fletcher_panics_total` | Counter | Total recovered panics in worker pool. |
