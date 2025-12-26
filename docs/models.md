# Model Support in Fletcher

Longbow Fletcher is designed to support multiple transformer-based embedding models. While it originated as a BERT-Tiny engine, it has been generalized to support more complex architectures like Nomic.

## Supported Models

| Model                     | Dimensions | Layers | Heads | Max Seq Len | Special Features             |
|---------------------------|------------|--------|-------|-------------|------------------------------|
| **BERT-Tiny**             | 128        | 2      | 2     | 512         | Absolute Position Embeddings |
| **Nomic-Embed-Text-v1.5** | 768        | 12     | 12    | 8192        | RoPE, SwiGLU                |

## Selecting a Model

Use the `--model` flag to specify which architecture to use. This automatically configures the internal dimensions and activation functions.

```bash
# Default (BERT-Tiny)
./fletcher --vocab vocab.txt --text "Hello world"

# Nomic-Embed-Text
./fletcher --model nomic-embed-text --vocab vocab.txt --weights nomic.bin --text "Hello world"
```

## Architectural Features

### 1. Rotary Positional Embeddings (RoPE)

Fletcher supports RoPE, which is required for modern models like Nomic. This enables the model to handle significantly longer context windows (up to 8192 tokens) more effectively than absolute positional embeddings.

### 2. SwiGLU Activation

Support for the SwiGLU activation function is included, providing better performance and accuracy for models trained with this architecture. On Metal backends, this is implemented as a fused kernel for maximum efficiency.

### 3. Absolute Positional Embeddings

Fletcher maintains support for high-efficiency absolute positional embeddings used by the BERT family of models.

## Adding New Models

Fletcher's `BertModel` architecture is generalized. To support a new model:

1. Define a new `BertConfig` in `internal/embeddings/model/bert.go`.
2. Add the model type to the `NewEmbedder` factory in `internal/embeddings/embeddings.go`.
3. Ensure necessary kernels (e.g., custom activations) are implemented in the CPU and Metal backends.
