# Fletcher Roadmap: Post-v0.2.0

**Status**: Optimization Phase Complete (2x Speedup achieved).
**Goal**: Enterprise Readiness & Scalability.

## 1. Release Management (RC Testing)

- **Subtask**: Cut `release/v0.2.0-rc1` branch.
- **Subtask**: Validate Helm chart deployment in Staging.
- **Subtask**: Run long-soak tests (24h) to check for memory leaks.
- **Subtask**: Publish official Docker images to GHCR.

## 2. Advanced Kernel Fusion (Metal)

- **Subtask**: Implement fused `Add + LayerNorm` kernel (FP32/FP16).
- **Subtask**: Implement fused `GEMM + GELU + Bias` kernel.
- **Subtask**: Profile dispatch overhead reduction (Goal: >2000 TPS).
- **Subtask**: Optimize `FlashAttention` for variable sequence lengths (paged attention).

## 1.1 Performance & Coherence (Next 10 Steps)

1. **Paged Attention Implementation**: Implement a Paged KV Cache to eliminate memory fragmentation and allow for much larger batch sizes.
2. **Bit-Level Q4_K Optimization**: Refactor Q4_K dequantization to use bit-field extracts, reducing instruction count in the inner loop by ~15%.
3. **Cross-Layer Float32 Residuals**: Transition residual connections to remain in FP32 throughout the network to prevent error accumulation across 12+ layers.
4. **Asynchronous KV-Cache Prefetch**: Pipeline KV-cache loading with the current layer's MatMul computation to hide memory latency.
5. **Softmax Log-Space Normalization**: Implement more stable log-softmax kernels to prevent overflow in deep models with large scale factors.
6. **Fused RMSNorm + Rotary Post-Processing**: Combine RMSNorm and Rotary Embeddings into a single kernel to save global memory bandwidth.
7. **Dynamic LoRA Adapter Support**: Implement a fused linear kernel that can apply LoRA weights with near-zero overhead.
8. **Adaptive Temperature Slicing**: Implement per-head scaling in Softmax to handle varying activation magnitudes across different attention heads.
9. **Predictive KV Cache Pruning**: Experiment with pruning unimportant KV tokens based on attention weight history to save cache space.
10. **SIMD-Accelerated Tokenization**: Use SIMD (NEON/AVX) for the BPE merge loop to resolve the bottleneck in long-context processing.

## 3. Quantization Support

- **Subtask**: Implement Q4_0 and Q8_0 dequantization kernels for Metal.
- **Subtask**: Update `loader.go` to handle quantized SafeTensors weights.
- **Subtask**: Add CLI flag `-quantization int8` / `int4`.
- **Subtask**: Validate accuracy degradation vs performance gain.

## 4. Cross-Platform Acceleration

- **Subtask**: Abstraction Layer Refactor (`Backend` interface decoupling).
- **Subtask**: Implement CUDA backend (via CGO/cuBLAS or Tritonserver).
- **Subtask**: Implement Vulkan/ROCm backend (via Kompute or similar).
- **Subtask**: Add biological CPU fallback (AVX-512 optimized).

## 5. Horizontal Scalability

- **Subtask**: Implement Fletcher-aware Load Balancer (consistent hashing on input).
- **Subtask**: Redis-backed Distributed Cache for embeddings.
- **Subtask**: Kubernetes HPA rules based on `fletcher_gpu_utilization`.
- **Subtask**: Leader election for cluster coordination (etcd).

## 6. Observability & Tracing

- **Subtask**: Integrate OpenTelemetry automated instrumentation.
- **Subtask**: Create Grafana Dashboard templates for exported metrics.
- **Subtask**: Add structured logging with correlation IDs.
- **Subtask**: Implement sampling profiler (pprof) endpoint security.

## 7. Model Ecosystem Expansion

- **Subtask**: Add support for `bge-m3` (multilingual).
- **Subtask**: Add support for `e5-mistral` (instruction-tuned).
- **Subtask**: Implement Reranker model architecture (Cross-Encoder).
- **Subtask**: Dynamic model loading/unloading API.

## 8. Security hardening

- **Subtask**: Implement mTLS for gRPC/Arrow Flight transport.
- **Subtask**: Add API Key authentication middleware.
- **Subtask**: Run fuzzing on `LoadFromSafeTensors` input parser.
- **Subtask**: Audit CGO memory safety boundaries.

## 9. Developer Experience (DX)

- **Subtask**: Create a Python SDK (`pip install longbow-fletcher`).
- **Subtask**: Create a Node.js client library.
- **Subtask**: Publish standard OpenAPI (Swagger) spec.
- **Subtask**: Write "Zero to Production" tutorial series.

## 10. Community & Integration

- **Subtask**: Add LangChain integration provider.
- **Subtask**: Add LlamaIndex vector store integration.
- **Subtask**: Create Hugging Face Spaces demo.
- **Subtask**: Setup GitHub Actions for automated nightly builds.
