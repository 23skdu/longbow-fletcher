# Fletcher Features

**Longbow Fletcher** is a pure Go application for generating vector text embeddings.

## Core Capabilities

- **Pure Go Embedding Engine**: Custom BERT-style Transformer implementation using `gonum` for linear algebra. No Python or CGO required.
- **WordPiece Tokenizer**: Compatible with standard BERT vocabulary files.
- **Apache Arrow Integration**: Formats embeddings as high-performance Arrow RecordBatches.
- **Apache Flight Client**: Sends data to Longbow servers using the Flight RPC protocol.

## Developer Features

- **Lorem Ipsum Generator**: Built-in test data generation for stress testing.
- **Configurable CLI**: All options exposed via command-line flags.
- **Minimal Docker Image**: Multi-stage build produces a ~10MB `scratch` image.

## Future Roadmap

- **SIMD Optimizations**: AVX2/NEON kernels for matrix multiplication.
- **Pre-trained Weights**: Support for loading standard HuggingFace checkpoints.
