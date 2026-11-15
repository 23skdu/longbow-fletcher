# Fletcher Integration Roadmap (Longbow v0.2.5+)

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

- [ ] 2.1.1 Send a valid `FlightDescriptor` with `Path = [datasetName]` on every `DoPut`
      (already set via `writer.SetFlightDescriptor`).
- [ ] 2.1.2 Ensure Fletcher's output schema (`text: utf8`, `embedding: fixed_size_list<float32>[dim]`)
      is accepted by Longbow's schema evolution (`Evolve`) – verify no metadata mismatches.
- [ ] 2.1.3 Add optional row identity column (`row_id: uint64`) to match Longbow's
      provenance expectations and enable idempotent upserts.
- [ ] 2.1.4 Support Longbow namespaces via `namespace/dataset` path parsing
      (`ParseNamespacedPath`).
- [ ] 2.1.5 Add backpressure handling for Longbow admission control responses.

### 2.2 Embedding Service Path (Longbow -> Fletcher)

Fletcher's Flight server should act as a first-class embedding provider for Longbow,
analogous to ONNX/WASM/Ollama providers.

- [ ] 2.2.1 Define a stable `DoExchange` protocol:
      - Input schema: `{row_id: uint64, text: utf8}` (and optional `modality`, `blob`).
      - Output schema: `{row_id: uint64, embedding: fixed_size_list<float32>[dim]}`.
- [ ] 2.2.2 Route requests by `FlightDescriptor`:
      - `Cmd = "Embed"` or `Path = ["embed", model]` for embedding requests.
      - Preserve Longbow's `peekedStream` first-message handling.
- [ ] 2.2.3 Support streaming partial results and per-batch error reporting.
- [ ] 2.2.4 Advertise supported models/dimensions via a `DoAction` (e.g. `GetModels`).

### 2.3 Version Skew

- [ ] 2.3.1 Keep Arrow Go, gRPC, and protobuf versions pinned to Longbow's
      (`v18.8.0`, `v1.85.0`) to avoid IPC wire incompatibilities.
- [ ] 2.3.2 Add a CI check that builds Fletcher against Longbow's module versions.

## 3. Authentication, Authorization, and Transport Security

Longbow v0.2.5 enforces RBAC (admin/writer/reader/tenant-admin), API keys, tenant
isolation, and TLS/mTLS.

- [ ] 3.1 Implement TLS/mTLS in `StartFlightServer` (currently a placeholder).
- [ ] 3.2 Add TLS options to `NewFlightClient`/`NewEmbeddingClient`.
- [ ] 3.3 Propagate API keys via gRPC metadata (`authorization: Bearer <key>`).
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

## Success Criteria

Integration is complete when:

1. Fletcher can embed a batch of text and ingest into Longbow with a single command.
2. Longbow can call Fletcher as a remote embedding provider over Arrow Flight.
3. TLS/mTLS and API-key auth work end-to-end.
4. Multi-modal (text + image) embeddings round-trip through Longbow.
5. CI builds Fletcher and Longbow together without version skew.

---

## Consolidated Legacy Roadmap (from nextsteps.md)

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
