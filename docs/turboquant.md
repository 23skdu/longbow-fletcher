# TurboQuant - Advanced Quantization for AI Embeddings

## Overview

TurboQuant is an advanced quantization algorithm developed by Google Research that achieves up to 6x memory reduction with zero accuracy loss. This implementation brings TurboQuant, PolarQuant, and QJL (Quantized Johnson-Lindenstrauss) to longbow-fletcher.

## Algorithm Components

### 1. TurboQuant

TurboQuant combines two key techniques:
- **PolarQuant**: Converts Cartesian coordinates to polar coordinates (radius + angles) for efficient compression
- **QJL**: Uses 1-bit Johnson-Lindenstrauss transform on residuals to eliminate bias

### 2. PolarQuant

Converts vector coordinates from Cartesian (x, y, z) to polar (radius, angles):
- **Radius**: Magnitude of the vector
- **Angles**: Directional components

This approach eliminates memory overhead by mapping data onto a fixed circular grid.

### 3. QJL (Quantized Johnson-Lindenstrauss)

Uses mathematical projection to:
- Reduce complex high-dimensional data
- Apply 1-bit sign quantization
- Use special estimators to maintain accuracy

## Usage

### Basic Quantization

```go
import "github.com/23skdu/longbow-fletcher/internal/device"

// Create data
data := make([]float32, 1024)
// ... fill data ...

// Configure TurboQuant
config := device.TurboQuantConfig{
    BitWidth:      4,   // 4-bit quantization
    BlockSize:     64,   // 64 elements per block
    UseQJL:        true, // Apply QJL correction
    QJLDimensions: 64,    // QJL projection dimensions
}

// Quantize
quantized := device.QuantizeTurbo(data, 1024, config)

// Decompress
decompressed := quantized.ToFloat32()

// Check compression ratio
ratio := quantized.CompressionRatio()
fmt.Printf("Compression ratio: %.2fx\n", ratio)
```

### Using PolarQuant Directly

```go
// Create PolarQuantizer for simpler use cases
polar := device.NewPolarQuantizer(64, 4) // blockSize=64, bitWidth=4

// Quantize
quantized, err := polar.Quantize(data)

// Decompress
decompressed := polar.Dequantize(quantized)
```

### Using QJL for Dimensionality Reduction

```go
// Create QJL quantizer
qjl := device.NewQJLQuantizer(64) // project to 64 dimensions

// Compress
compressed := qjl.Compress(data)

// Decompress
reconstructed := qjl.Decompress(compressed)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| BitWidth | int | 4 | Quantization bit width (2-8) |
| BlockSize | int | 64 | Elements per quantization block |
| UseQJL | bool | false | Enable QJL residual correction |
| QJLDimensions | int | 64 | Dimensions for QJL projection |

## Performance Characteristics

- **Memory Reduction**: Up to 6x (4-bit quantization)
- **Speed**: Up to 8x faster attention computation
- **Accuracy**: Zero accuracy loss with QJL enabled

## Integration with Embeddings

The quantization can be applied to embedding vectors before storage or transmission:

```go
// Quantize embedding output
embeddings := embedder.ProxyEmbedBatch(ctx, texts)
quantized := device.QuantizeTurbo(embeddings, len(embeddings), config)

// Store or transmit quantized data
storage.Save(quantized.data)

// Later: decompress
decompressed := quantized.ToFloat32()
```

## Benchmarking

Run the built-in benchmarks:

```bash
go test -bench=BenchmarkTurboQuant ./internal/device/...
```

## References

- [TurboQuant: Redefining AI efficiency with extreme compression](https://arxiv.org/abs/2504.19874) - Google Research
- [Quantized Johnson-Lindenstrauss](https://arxiv.org/abs/2406.03482)
- [PolarQuant](https://arxiv.org/abs/2502.02617)

## Notes

- TurboQuant is particularly effective for KV cache compression in LLMs
- QJL correction provides near-zero accuracy loss
- The algorithm is data-oblivious (works without dataset-specific tuning)
