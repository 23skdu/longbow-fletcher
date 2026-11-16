# Longbow Fletcher Integration Roadmap (Longbow v0.2.5+)

This roadmap tracks the work required for Fletcher to integrate cleanly with the
latest [Longbow](https://github.com/23skdu/longbow) vector database release
(v0.2.5, Go 1.27.4). It supersedes the legacy `nextsteps.md` planning notes and
is the single source of truth for cross-project compatibility work.

## 1. Deep Analysis Summary

Fletcher is a pure-Go transformer embedding engine with three execution modes:

1. **CLI batch mode** (`cmd/fletcher`) – tokenizes and embeds text, writes Arrow IPC.
2. **HTTP server mode** (`-listen`) – concurrent request batching with admission control.
3. **Arrow Flight server mode** (`-flight`) – bidirectional `DoExchange` embedding service.

Core components:

- `internal/embeddings.Embedder` orchestrates tokenization, model forward pass, pooling,
  caching, and device admission control.
- `internal/embeddings/model.BertModel` implements BERT-style transformers with support
  for RoPE, SwiGLU, fused QKV, and multiple pooling strategies.
- `internal/device` provides CPU (BLAS), Metal (Apple Silicon), and CUDA backends behind
  a common `Backend` interface.
- `internal/client` provides `FlightClient`/`EmbeddingClient` for Arrow Flight transport
  to and from Longbow.
- `internal/embeddings/tokenizer` implements WordPiece with SIMD helpers.
- `internal/embeddings/weights` loads SafeTensors checkpoints.

Supported models: BERT-Tiny, all-MiniLM-L6-v2, nomic-embed-text, RoBERTa, XLM-RoBERTa,
bge-m3, e5-mistral-7b, plus ViT/CLIP vision and SPLADE sparse encoders (partial).

## 2. Arrow Flight Protocol Compatibility

Longbow v0.2.5 exposes `DoPut`, `DoGet`, `DoExchange`, and `DoAction` over Arrow
Flight with RBAC authorization and TLS/mTLS. Fletcher must match these contracts.

### 2.1 Ingest Path (Fletcher -> Longbow)

- [ ] 2.1.2 Ensure Fletcher's output schema (`text: utf8`, `embedding: fixed_size_list<float32>[dim]`)
      is accepted by Longbow's schema evolution (`Evolve`) – verify no metadata mismatches.
- [ ] 2.1.4 Support Longbow namespaces via `namespace/dataset` path parsing
      (`ParseNamespacedPath`).
- [ ] 2.1.5 Add backpressure handling for Longbow admission control responses.

### 2.2 Embedding Service Path (Longbow -> Fletcher)

Fletcher's Flight server should act as a first-class embedding provider for Longbow,
analogous to ONNX/WASM/Ollama providers.

- [ ] 2.2.2 Route requests by `FlightDescriptor`:
      - `Cmd = "Embed"` or `Path = ["embed", model]` for embedding requests.
      - Preserve Longbow's `peekedStream` first-message handling.
- [ ] 2.2.3 Support streaming partial results and per-batch error reporting.
- [ ] 2.2.4 Advertise supported models/dimensions via a `DoAction` (e.g. `GetModels`).

### 2.3 Version Skew

- [ ] 2.3.2 Add a CI check that builds Fletcher against Longbow's module versions.

## 3. Authentication, Authorization, and Transport Security

Longbow v0.2.5 enforces RBAC (admin/writer/reader/tenant-admin), API keys, tenant
isolation, and TLS/mTLS.

- [ ] 3.4 Support tenant/namespace scoping in client requests.
- [ ] 3.5 Add audit-friendly structured logs with correlation IDs.

## 4. Multi-Modal Alignment

Longbow supports image/audio embedding pipelines (CLIP/OpenCLIP, Whisper/CLAP) and
cross-modal search.

- [ ] 4.1 Complete `ClipVisionEncoder.EncodeImage` (currently `ErrNotImplemented`).
- [ ] 4.2 Add audio encoder path (Whisper/CLAP) or document Fletcher's text-only scope.
- [ ] 4.3 Align Fletcher's image preprocessing with Longbow's blob schema
      (`blob + modality + vector`).
- [ ] 4.4 Expose modality in Flight `DoExchange` input/output schemas.

## 5. Model and Dimension Compatibility

Longbow's learned-index features track `embedding_provider` and `embedding_model`.

- [ ] 5.1 Register `"fletcher"` as an embedding provider in Longbow's
      `embeddingProviderOrdinal` (upstream change).
- [ ] 5.2 Document supported `(model, dimension)` pairs and ensure Longbow dataset
      creation uses matching dimensions.
- [ ] 5.3 Add model hot-swap / multi-model serving endpoint.
- [ ] 5.4 Validate FP16 transport (`-transport-fmt fp16`) against Longbow's
      vector storage expectations.

## 6. Observability

Longbow uses Prometheus metrics and OpenTelemetry tracing with defined naming.

- [ ] 6.1 Align Fletcher metric names with Longbow conventions
      (`fletcher_embedding_*`).
- [ ] 6.2 Propagate trace context over Flight gRPC metadata.
- [ ] 6.3 Add `/metrics` and pprof endpoints to HTTP server mode.
- [ ] 6.4 Provide Grafana dashboard templates for Fletcher.

## 7. Testing and Validation

- [ ] 7.1 Add integration tests that spin up Longbow and Fletcher and verify
      end-to-end ingest + search.
- [ ] 7.2 Add Arrow Flight round-trip tests against Longbow's `DoPut`/`DoGet`.
- [ ] 7.3 Add RBAC/TLS integration tests.
- [ ] 7.4 Add benchmark comparisons using Longbow's benchmark harness.
- [ ] 7.5 Add fuzz tests for Flight input schemas.

## 8. Release and Documentation

- [ ] 8.1 Update Dockerfiles to Go 1.27.4 and aligned module cache.
- [ ] 8.2 Publish Fletcher as an embedding provider in Longbow docs.
- [ ] 8.3 Add "Zero to Production" tutorial for Fletcher + Longbow.
- [ ] 8.4 Cut `v0.3.0` release with aligned dependencies.

## 9. Batched Implementation Plan (Subtasks)

### Batch A - Protocol and Transport (High Impact)
- [ ] A.1 Add integration test harness skeleton under `tests/integration/`.
- [ ] A.2 Implement Longbow namespace path parsing in client (`namespace/dataset`).
- [ ] A.3 Add admission-control backpressure retries in `AsyncStream`.
- [ ] A.4 Route Flight `DoExchange` by descriptor (`embed`, `search`, `ingest`).
- [ ] A.5 Implement per-batch error reporting in `DoExchange` responses.
- [ ] A.6 Add `GetModels`/`GetDimensions` DoAction responses.
- [ ] A.7 Add CI job building against Longbow module versions.

### Batch B - Security and Observability (High Impact)
- [ ] B.1 Implement full mTLS client/server certificate loading.
- [ ] B.2 Add tenant propagation to all client requests.
- [ ] B.3 Add correlation IDs to HTTP and Flight logs.
- [ ] B.4 Align metric names and add Grafana dashboard templates.
- [ ] B.5 Propagate OpenTelemetry trace context over gRPC metadata.

### Batch C - Multi-Modal (Medium Impact)
- [ ] C.1 Complete CLIP vision encoder with weights loading.
- [ ] C.2 Add image preprocessing pipeline and tests.
- [ ] C.3 Align multi-modal schema with Longbow (`blob + modality + vector`).
- [ ] C.4 Add audio encoder (Whisper/CLAP) or document text-only scope.
- [ ] C.5 Expose modality in Flight input/output schemas.

### Batch D - Quantization and TurboQuant (Medium Impact)
- [ ] D.1 Implement Metal Q4_0/Q8_0 dequantization kernels.
- [ ] D.2 Implement CUDA Q4_0/Q8_0 dequantization kernels.
- [ ] D.3 Add `--quantization` CLI flags and validation.
- [ ] D.4 Add quantization accuracy vs performance tests.
- [ ] D.5 Finish TurboQuant polar/QJL kernels and tests.
- [ ] D.6 Wire quantized tensors into model loading path.

### Batch E - Advanced Caching and Model Ecosystem (Medium Impact)
- [ ] E.1 Wire `PrefixCache` into `Embedder` with token-based keys.
- [ ] E.2 Add chunked prefill support for long sequences.
- [ ] E.3 Add LoRA adapter application path.
- [ ] E.4 Add model hot-swap / multi-model serving endpoint.
- [ ] E.5 Document supported model/dimension matrix.

### Batch F - Testing and Release (High Impact)
- [ ] F.1 Add end-to-end integration tests with Longbow.
- [ ] F.2 Add Flight round-trip tests.
- [ ] F.3 Add RBAC/TLS integration tests.
- [ ] F.4 Add Flight schema fuzz tests.
- [ ] F.5 Update Dockerfiles and publish v0.3.0.

## 10. Deep Analysis & Legacy TODOs (from docs/nextsteps.md)

## Executive Summary

Fletcher is a **production-ready** text embedding engine with solid Metal GPU acceleration and multi-model support. However, several **TODOs and incomplete implementations** block it from reaching parity with mainstream embedding tools like sentence-transformers, Ollama, and llama.cpp.

**Critical Finding**: The codebase has ~6 real TODOs (not counting conversion script utility TODOs), plus several unimplemented tensor operations in the CUDA backend that need completion for feature parity.

### 1.1 High-Priority TODOs (Core Engine)

| File | Line | Issue | Impact | Effort |
|------|------|-------|--------|--------|
| `internal/device/cuda_linux.go` | 20 | `GetVRAMUsage()` not implemented | Medium | Low |
| `internal/device/cuda_linux.go` | 73 | `Cast()` kernel for FP16<->FP32 | High | Medium |
| `internal/device/cuda_linux.go` | 120+ | Missing tensor ops: `Add`, `AddScalar`, `Scale`, `Tanh`, `Slice`, `Paste` | High | Medium |
| `internal/embeddings/embeddings.go` | 670 | Weight preloading commented out | Medium | Low |
| `internal/embeddings/model/pool.go` | 13 | `GetSeqHidden()` returns nil (stub) | Low | Low |
| `internal/embeddings/model/pool.go` | 29 | `PutSeqHidden()` is no-op (stub) | Low | Low |

### 1.2 Missing Tensor Operations (CUDA Backend)

The CUDA backend declares these methods but they are not implemented:

- `Add`, `AddScalar`, `Scale`, `Tanh`, `Slice`, `Paste`, `Cast`, `GetVRAMUsage`

**Status**: CUDA backend currently falls back to CPU for these operations.

## 2. Feature Gaps vs Similar Tools

### 2.1 Quantization Support

| Feature | Fletcher | sentence-transformers | Ollama | llama.cpp |
|---------|----------|----------------------|--------|-----------|
| FP16 | ✅ Metal/CPU | ✅ | ✅ | ✅ |
| INT8 | ❌ | ✅ | ✅ | ✅ |
| INT4 (Q4_0/Q4_K) | ❌ | ❌ | ✅ | ✅ |
| Q8_0 | ❌ | ❌ | ✅ | ✅ |

### 2.2 Model Ecosystem

| Model Type | Fletcher | Priority |
|------------|----------|----------|
| BERT (all variants) | ✅ | - |
| RoBERTa/XLM-RoBERTa | ✅ | - |
| Nomic-Embed-Text | ✅ | - |
| BGE-M3 | ✅ | - |
| E5-Mistral | ✅ | - |
| **GTE (General Text Embeddings)** | ❌ | High |
| **E5-v2** | ❌ | Medium |
| **bge-m3** | ✅ | - |
| **Custom GGUF models** | ❌ | Low |

### 2.3 Production Features

| Feature | Fletcher | Status |
|---------|----------|--------|
| Dynamic Batching | ✅ | Implemented |
| Admission Control | ✅ | Implemented |
| OpenTelemetry | ✅ | Basic |
| mTLS | ❌ | Not implemented |
| API Key Auth | ✅ | Implemented (HTTP) |
| Rate Limiting | ✅ | Implemented |
| Prometheus Metrics | ✅ | Implemented |
| Health Checks | ✅ | Implemented |
| Graceful Shutdown | ⚠️ | Partial |

### 2.4 Observability

| Feature | Fletcher | Status |
|---------|----------|--------|
| Structured Logging | ✅ | zerolog |
| Distributed Tracing | ⚠️ | Basic OTel |
| Metrics Export | ✅ | Prometheus |
| Profiling Endpoints | ✅ | pprof |
| Grafana Dashboards | ❌ | Not provided |

## 3. Recommended Next Steps

### Phase 1: Fix Critical TODOs (1-2 weeks)

1. **CUDA Backend Completion**
   - Implement `Cast()` kernel (FP16<->FP32)
   - Implement `Add`, `AddScalar`, `Scale`, `Tanh` kernels
   - Implement `Slice` for variable sequences
   - Implement `GetVRAMUsage()` using `cudaMemGetInfo`
   - Uncomment weight preloading in embeddings.go

2. **Pool Buffer Implementation**
   - Fix `GetSeqHidden()`/`PutSeqHidden()` for proper pooling
   - Or remove if unused

### Phase 2: Quantization (2-3 weeks)

1. **INT8 Quantization**
   - Add quantization kernels for Metal/CUDA
   - Update loader.go for quantized weights
   - Add `--quantization int8` CLI flag
   - Validate accuracy (<2% degradation)

2. **INT4/Q4_0 Support**
   - Implement Q4_0 dequantization (Metal + CUDA)
   - Add Q4_K support for better quality
   - Benchmark memory/accuracy tradeoffs

### Phase 3: Model Expansion (2 weeks)

1. Add GTE (General Text Embeddings) support
2. Add E5-v2 support
3. Implement GGUF loader (optional)

### Phase 4: Production Hardening (2-3 weeks)

1. Implement mTLS for gRPC/Flight
2. Add graceful shutdown handling
3. Create Grafana dashboard templates
4. Add comprehensive integration tests

### Phase 5: Observability (1 week)

1. Complete OpenTelemetry instrumentation
2. Add correlation IDs to logs
3. Create alerting rules

## 4. Priority Matrix

| Task | Impact | Effort | Priority |
|------|--------|--------|----------|
| CUDA Cast kernel | High | Medium | P0 |
| CUDA tensor ops | High | Medium | P0 |
| Weight preloading | Medium | Low | P1 |
| INT8 quantization | High | Medium | P1 |
| GTE model support | Medium | Low | P2 |
| mTLS | Medium | Medium | P2 |
| Grafana dashboards | Low | Low | P3 |

### Top 5 Immediate Actions

1. Fix CUDA `Cast()` kernel (blocks FP16 on GPU)
2. Implement CUDA `Add`/`Scale`/`Tanh` (blocks model inference)
3. Uncomment weight preloading (performance)
4. Fix `GetSeqHidden`/`PutSeqHidden` (correctness)
5. Add INT8 quantization (memory efficiency)

## 5. Comparison with Competitors

### Performance (Apple M3 Pro, BERT-Tiny)

| Metric | Fletcher (Metal) | PyTorch (MPS) | Ollama |
|--------|------------------|---------------|--------|
| Peak Throughput | 24,200 vec/s | 14,800 vec/s | ~5,000 vec/s |
| Single Latency | 0.48 ms | 4.77 ms | ~2 ms |
| Memory | <10 MB | ~370 MB | ~500 MB |

### Feature Comparison

| Feature | Fletcher | sentence-transformers | Ollama |
|---------|----------|----------------------|--------|
| Pure Go | ✅ | ❌ (Python) | ❌ (Go+C++) |
| Metal GPU | ✅ | ⚠️ (MPS) | ❌ |
| CUDA | ⚠️ (Partial) | ✅ | ✅ |
| Quantization | ❌ | ✅ | ✅ |
| Multi-model | ✅ | ✅ | ✅ |
| REST API | ✅ | ❌ | ✅ |
| Arrow Flight | ✅ | ❌ | ❌ |
| Longbow Integration | ✅ | ❌ | ❌ |

## 6. Files Requiring Changes

### Core Engine Changes

- `internal/device/cuda_linux.go` - Implement missing kernels
- `internal/device/cuda_linux.cu` - Add CUDA kernel implementations
- `internal/embeddings/embeddings.go` - Uncomment weight preloading
- `internal/embeddings/model/pool.go` - Fix buffer pool
- `internal/device/quantization.go` - Add quantization support

### New Files

- `internal/device/quantization_cuda.go` - CUDA quantization
- `internal/device/quantization_metal.go` - Metal quantization
- `internal/models/gte.go` - GTE model implementation
- `internal/models/e5.go` - E5 model implementation

## 7. Success Criteria

Fletcher reaches parity when:

1. ✅ Metal GPU achieves >20k vec/s (DONE)
2. ⚠️ CUDA backend fully functional (Partial)
3. ❌ INT8 quantization available
4. ❌ Multi-model support (GTE, E5)
5. ❌ Full production features (mTLS, graceful shutdown)
6. ✅ Longbow integration (Arrow Flight)

## 8. Mac Metal Performance Improvement Roadmap (Implementation Phase)

### 8.1 Current Performance Analysis

- **Peak Throughput**: ~24,200 vec/s (Metal, BERT-Tiny, batch 32)
- **Single Latency**: ~0.48 ms
- **Memory**: <10 MB

### 8.2 Implementation Plan

1. **Fused Kernels**
   - Fused Add+LayerNorm
   - Fused GEMM+GELU+Bias
   - FlashAttention for variable lengths

2. **Memory Optimization**
   - Buffer pooling
   - In-place operations
   - KV cache optimization

3. **Dispatch Optimization**
   - Reduce kernel launch overhead
   - Batch small operations

### 8.3 Detailed Task Breakdown

- [ ] Implement fused Add+LayerNorm kernel
- [ ] Implement fused GEMM+GELU+Bias kernel
- [ ] Optimize FlashAttention for variable lengths
- [ ] Reduce dispatch overhead
- [ ] Profile and optimize hot paths

### 8.4 Test Requirements

- [ ] Unit tests for each fused kernel
- [ ] Performance benchmarks
- [ ] Accuracy validation vs reference

### 8.5 Success Metrics

- Target: >30k vec/s peak throughput
- Target: <0.3 ms single latency
- Target: <5 MB memory footprint

## 9. SSH Access & Remote Execution

### ancalagon Connection

- SSH key: `~/.ssh/ancalagon_key`
- Host: `ancalagon` (configured in ~/.ssh/config)
- Purpose: CUDA testing and development

### Remote Workflow

1. SSH to ancalagon
2. Pull latest code
3. Build and test CUDA backend
4. Run benchmarks
5. Push results

## 10. File Changes Summary

### Modified Files

- `internal/device/cuda_linux.go` - Implement missing ops
- `internal/device/cuda_linux.cu` - Add CUDA kernels
- `internal/embeddings/embeddings.go` - Uncomment preloading
- `internal/embeddings/model/pool.go` - Fix buffer pool
- `internal/device/quantization.go` - Add quantization

### New Files

- `internal/device/quantization_cuda.go`
- `internal/device/quantization_metal.go`
- `internal/models/gte.go`
- `internal/models/e5.go`

## 11. Implementation Checklist

- [ ] CUDA Cast kernel
- [ ] CUDA Add/Scale/Tanh kernels
- [ ] CUDA Slice/Paste
- [ ] CUDA GetVRAMUsage
- [ ] Weight preloading
- [ ] Buffer pool fix
- [ ] INT8 quantization
- [ ] Q4_0 quantization
- [ ] GTE model
- [ ] E5 model
- [ ] mTLS
- [ ] Grafana dashboards

## 12. Performance Comparison Findings (2026-03-24)

### Test Results Summary

- Fletcher Metal: 24,200 vec/s peak
- PyTorch MPS: 14,800 vec/s peak
- Speedup: 1.6x peak, 2.5x sustained

### Key Issues Identified

1. Small sequence lengths cause MPS crashes (fixed)
2. CUDA backend incomplete (fallback to CPU)
3. Model compatibility limited to BERT-style

### Performance Baseline (CPU)

- BERT-Tiny: ~1,200 vec/s (CPU, 4 cores)
- Memory: ~5 MB

### Test Results (2026-03-24)

- All unit tests passing
- Metal benchmarks stable
- CUDA tests skipped (incomplete)

## 13. Phase 2: Production Readiness Plan

### Part 1: Fix Model Weights Pipeline (Priority: Critical)

- [ ] Audit loader.go for format support
- [ ] Add SafeTensors validation
- [ ] Implement weight preloading
- [ ] Add model caching

### Part 2: Implement Real CUDA GPU Kernels (Priority: Critical)

- [ ] Implement all missing tensor ops
- [ ] Add CUDA memory management
- [ ] Implement FP16 support
- [ ] Add CUDA benchmarks

### Part 3: Metal Performance Optimization (Priority: High)

- [ ] Implement fused kernels
- [ ] Optimize dispatch overhead
- [ ] Add profiling tools

### Part 4: Ollama Comparison Tests (Priority: High)

- [ ] Set up Ollama on ancalagon
- [ ] Create comparison benchmarks
- [ ] Validate accuracy

### Part 5: Quantization Support (Priority: Medium)

- [ ] INT8 quantization
- [ ] INT4 quantization
- [ ] Accuracy validation

### Part 6: API & Auth (Priority: Medium)

- [ ] API key authentication
- [ ] Rate limiting (done)
- [ ] mTLS

### Part 7: Model Support Expansion (Priority: Medium)

- [ ] GTE support
- [ ] E5 support

### Part 8: Observability (Priority: Low)

- [ ] Grafana dashboards
- [ ] Alerting rules

### Part 9: Client SDKs (Priority: Low)

- [ ] Python SDK
- [ ] Node.js SDK

### Part 10: CI/CD & Testing (Priority: Low)

- [ ] GitHub Actions
- [ ] Nightly builds

## 14. vLLM Feature Parity Roadmap

### vLLM Key Features (What's Missing in Fletcher)

- PagedAttention
- Continuous batching
- Tensor parallelism
- Pipeline parallelism
- OpenAI-compatible API
- LoRA adapter support
- Speculative decoding
- Multi-modal support

### Part 11: Multi-Modal Support (P0)

- [ ] Image embeddings (CLIP)
- [ ] Audio embeddings (Whisper/CLAP)
- [ ] Cross-modal search

### Part 12: Sparse Embeddings (P1)

- [ ] SPLADE support
- [ ] Sparse vector storage
- [ ] Hybrid search optimization

### Part 13: Model Ecosystem Expansion (P1)

- [ ] GTE models
- [ ] E5 models
- [ ] BGE models
- [ ] Custom model loading

### Part 14: Advanced Features (P2)

- [ ] Query rewriting
- [ ] Reranking
- [ ] Multi-vector representations

### Part 15: OpenAI Compatibility (P1)

- [ ] OpenAI-compatible API
- [ ] Streaming responses
- [ ] Function calling

### Part 16: Hardware Support (P3)

- [ ] AMD ROCm
- [ ] Intel Arc
- [ ] Vulkan

### Feature Parity Priority Matrix

| Feature | Priority | Effort | Status |
|---------|----------|--------|--------|
| PagedAttention | P0 | High | Not started |
| Multi-modal | P0 | High | Partial (CLIP stub) |
| Continuous batching | P1 | Medium | Partial |
| LoRA support | P1 | Medium | Not started |
| OpenAI API | P1 | Medium | Partial |
| Quantization | P1 | Medium | Not started |
| Sparse embeddings | P2 | Medium | Partial (SPLADE stub) |
| Tensor parallelism | P3 | High | Not started |

### Top 5 Priorities for vLLM Parity

1. Multi-modal support (image/audio)
2. Continuous batching optimization
3. LoRA adapter support
4. OpenAI-compatible API
5. Quantization (INT8/INT4)

## 15. vLLM Feature Parity Roadmap

### vLLM Key Features (What's Missing in Fletcher)

- PagedAttention
- Continuous batching
- Tensor parallelism
- Pipeline parallelism
- OpenAI-compatible API
- LoRA adapter support
- Speculative decoding
- Multi-modal support

### Part 11: Multi-Modal Support (P0)

- [ ] Image embeddings (CLIP)
- [ ] Audio embeddings (Whisper/CLAP)
- [ ] Cross-modal search

### Part 12: Sparse Embeddings (P1)

- [ ] SPLADE support
- [ ] Sparse vector storage
- [ ] Hybrid search optimization

### Part 13: Model Ecosystem Expansion (P1)

- [ ] GTE models
- [ ] E5 models
- [ ] BGE models
- [ ] Custom model loading

### Part 14: Advanced Features (P2)

- [ ] Query rewriting
- [ ] Reranking
- [ ] Multi-vector representations

### Part 15: OpenAI Compatibility (P1)

- [ ] OpenAI-compatible API
- [ ] Streaming responses
- [ ] Function calling

### Part 16: Hardware Support (P3)

- [ ] AMD ROCm
- [ ] Intel Arc
- [ ] Vulkan

### Feature Parity Priority Matrix

| Feature | Priority | Effort | Status |
|---------|----------|--------|--------|
| PagedAttention | P0 | High | Not started |
| Multi-modal | P0 | High | Partial (CLIP stub) |
| Continuous batching | P1 | Medium | Partial |
| LoRA support | P1 | Medium | Not started |
| OpenAI API | P1 | Medium | Partial |
| Quantization | P1 | Medium | Not started |
| Sparse embeddings | P2 | Medium | Partial (SPLADE stub) |
| Tensor parallelism | P3 | High | Not started |

### Top 5 Priorities for vLLM Parity

1. Multi-modal support (image/audio)
2. Continuous batching optimization
3. LoRA adapter support
4. OpenAI-compatible API
5. Quantization (INT8/INT4)