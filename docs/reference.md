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
| `-weights` | `bert_tiny.safetensors` | Path to weights file (.bin or .safetensors). |

### Server

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-listen` | `""` | HTTP server address (e.g., `:8080`). |
| `-flight` | `""` | Arrow Flight server address (e.g., `:9090`). |
| `-server` | `""` | Remote Longbow Flight address for forwarding. |

---

## Supported Models

| Model | Dimensions | Layers | Heads | Max Seq | Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT-Tiny** | 128 | 2 | 2 | 512 | Absolute Pos |
| **Nomic-v1.5** | 768 | 12 | 12 | 8192 | **RoPE**, **SwiGLU** |

### Special Features

- **RoPE**: Rotary Positional Embeddings allow for long contexts (8k+).
- **SwiGLU**: Activation function used in Nomic, accelerated by custom Metal kernels.

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
