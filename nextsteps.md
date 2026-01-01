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
