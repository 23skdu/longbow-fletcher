# Fletcher Features

**Longbow Fletcher** is a high-performance transformer-based embedding engine written in pure Go.

## Core Capabilities

- **Multi-Model Support**: Select between `bert-tiny` and `nomic-embed-text-v1.5` architectures.
- **Metal GPU Acceleration**: Leveraging Apple Silicon's GPU with hand-coded FP16 kernels for ~24,000 vec/s throughput.
- **Pure Go Inference**: Custom transformer implementation with optimized CPU backends (CGO/BLAS).
- **Modern Transformer Ops**: Support for **RoPE** (Rotary Positional Embeddings) and **SwiGLU** activation.
- **WordPiece Tokenizer**: Fully compatible with standard BERT subword vocabularies.
- **Apache Arrow & Flight**: Native integration for high-performance data transport.

## Developer Features

- **Lorem Ipsum Generator**: Built-in test data generation for stress testing.
- **Configurable CLI**: Full control over models, vocabs, weights, and hardware selection.
- **Minimal Footprint**: Multi-stage Docker builds produce a ~10MB scratch image.

## Roadmap

- **Quantization**: Support for 4-bit and 8-bit weight quantization.
- **Mean Pooling**: Alternative pooling strategies for sentence-level embeddings.
- **Clustering**: Built-in vector clustering for pre-indexing analysis.
