# Fletcher Embedding Architecture

Fletcher uses a custom pure Go implementation of a BERT-style Transformer for generating text embeddings.

## Architecture Overview

```
Text Input → Tokenizer → Embeddings → Encoder → Pooler → Vector Output
```

### Components

| Component | Description |
|-----------|-------------|
| **WordPiece Tokenizer** | Splits text into subword tokens using a BERT vocabulary |
| **Word Embeddings** | Lookup table mapping token IDs to vectors |
| **Position Embeddings** | Encodes sequential position information |
| **Transformer Encoder** | Stack of self-attention + feed-forward layers |
| **Pooler** | Extracts [CLS] token representation as final embedding |

## Weight Initialization

By default, Fletcher uses **Xavier/Glorot initialization** for all weight matrices:

```
limit = sqrt(6 / (fan_in + fan_out))
weight ~ Uniform(-limit, limit)
```

This produces meaningful non-zero embeddings without external weight files.

## Configuration

Default BERT-Tiny configuration:

| Parameter | Value |
|-----------|-------|
| Hidden Size | 128 |
| Layers | 2 |
| Attention Heads | 2 |
| Intermediate Size | 512 |
| Vocab Size | 30,522 |

## Performance

Fletcher is optimized for throughput using:

- `gonum/mat` for optimized linear algebra
- Pre-allocated buffers to minimize GC pressure
- Future: SIMD kernels for AMD64/ARM64
