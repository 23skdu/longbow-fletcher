# How to Enable GPU Acceleration

Fletcher natively supports Metal Performance Shaders (MPS) on Apple Silicon (M1/M2/M3), delivering significant performance improvements over CPU execution.

## Prerequisites

- **Hardware**: Mac with Apple Silicon.
- **OS**: macOS 12.0 (Monterey) or later recommended.
- **Build**: Ensure Fletcher was built with CGO enable (standard build).

## Enabling Metal

Add the `--gpu` flag to your command.

```bash
./bin/fletcher --gpu --vocab vocab.txt --text "Accelerated inference"
```

## Configuration

### FP16 Precision (Default)

By default, the Metal backend uses FP16 (Half Precision) for inference. This offers:

- **2x Throughput**: Utilizing specialized hardware units.
- **Lower Memory**: Halves VRAM usage.
- **High Accuracy**: Critical components like Softmax use FP32 accumulation.

To force FP32 (if needed for debugging):

```bash
./bin/fletcher --gpu --precision fp32 ...
```

## Performance Expectations

On an M3 Pro:

- **CPU**: ~10,000 vectors/sec
- **GPU (Metal)**: ~24,000 vectors/sec (**2.4x Speedup**)

## Troubleshooting

- **Crash on Start**: Ensure your macOS is up to date and you are not running in a simulated environment (Rosetta).
- **Zero Output**: Ensure you are using a compatible model configuration (BERT-tiny or Nomic).
