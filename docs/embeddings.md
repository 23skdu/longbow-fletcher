# Fletcher Embedding Architecture

Fletcher uses a custom pure Go implementation of a BERT-style Transformer for generating text embeddings, optimized for multiple model architectures.

## Architecture Overview

```text
Text Input → Tokenizer → Embeddings (Absolute/RoPE) → Encoder → Pooler → Vector Output
```

### Components

| Component | Description |
|-----------|-------------|
| **WordPiece Tokenizer** | Splits text into subword tokens using a BERT-style vocabulary. |
| **Word Embeddings** | Lookup table mapping token IDs to vectors. |
| **Positional Embeddings** | Supports both **Absolute** (BERT) and **Rotary (RoPE)** embeddings. |
| **Transformer Encoder** | Configurable stack of self-attention + feed-forward layers. Supports **SwiGLU** activation. |
| **Pooler** | Extracts [CLS] token or Mean Pool representation as final embedding. |

## Weight Initialization

By default (if no weights are provided), Fletcher uses **Xavier/Glorot initialization** for all weight matrices:

```text
limit = sqrt(6 / (fan_in + fan_out))
weight ~ Uniform(-limit, limit)
```

## Model Configurations

Fletcher dynamically adjusts its architecture based on the selected model:

| Parameter    | BERT-Tiny | Nomic v1.5 |
|--------------|-----------|------------|
| Hidden Size  | 128       | 768        |
| Layers       | 2         | 12         |
| Attention Heads | 2         | 12         |
| Intermediate | 512       | 3072       |
| Activation   | GELU      | SwiGLU     |
| Position     | Absolute  | RoPE       |

## Hardware Acceleration

- **CPU**: Parallelized execution using `gonum/mat` and custom SIMD-optimized loops.
- **GPU**: Native Metal implementation with Float16 support for maximum throughput on Apple Silicon.
