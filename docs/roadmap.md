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







      (already set via `writer.SetFlightDescriptor`).






      is accepted by Longbow's schema evolution (`Evolve`) – verify no metadata mismatches.



      provenance expectations and enable idempotent upserts.






      (`ParseNamespacedPath`).










### 2.2 Embedding Service Path (Longbow -> Fletcher)







Fletcher's Flight server should act as a first-class embedding provider for Longbow,



analogous to ONNX/WASM/Ollama providers.







      - Input schema: `{row_id: uint64, text: utf8}` (and optional `modality`, `blob`).



      - Output schema: `{row_id: uint64, embedding: fixed_size_list<float32>[dim]}`.






      - `Cmd = "Embed"` or `Path = ["embed", model]` for embedding requests.



      - Preserve Longbow's `peekedStream` first-message handling.












### 2.3 Version Skew







      (`v18.8.0`, `v1.85.0`) to avoid IPC wire incompatibilities.










## 3. Authentication, Authorization, and Transport Security







Longbow v0.2.5 enforces RBAC (admin/writer/reader/tenant-admin), API keys, tenant



isolation, and TLS/mTLS.










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











---







## Deep Analysis & Legacy TODOs (from docs/nextsteps.md)







**Analysis Date**: 2026-03-23  



**Purpose**: Identify TODO stubs/mock code and define roadmap to bring engine on par with similar tools (sentence-transformers, llama.cpp, etc.)







---







## Executive Summary







Fletcher is a **production-ready** text embedding engine with solid Metal GPU acceleration and multi-model support. However, several **TODOs and incomplete implementations** block it from reaching parity with mainstream embedding tools like sentence-transformers, Ollama, and llama.cpp.







**Critical Finding**: The codebase has ~6 real TODOs (not counting conversion script utility TODOs), plus several unimplemented tensor operations in the CUDA backend that need completion for feature parity.







---







## 1. TODO Stubs & Mock Code Found







### 1.1 High-Priority TODOs (Core Engine)







| File | Line | Issue | Impact | Effort |



|------|------|-------|--------|--------|



| `internal/device/cuda_linux.go` | 103 | `GetVRAMUsage()` returns `(0, 0)` - cudaMemGetInfo not implemented | No VRAM monitoring on CUDA | Low |



| `internal/device/cuda_linux.go` | 335-336 | `Cast()` not implemented - panics on FP16<->FP32 | CUDA FP16 mode broken for output | Medium |



| `internal/embeddings/model/bert.go` | 586 | Manual element-by-element copy instead of `Paste/SetSlice` | 10-15% slower attention | Medium |



| `internal/embeddings/metrics.go` | 40 | Load balance efficiency metric not implemented | No observability for load balancing | Low |



| `internal/embeddings/embeddings.go` | 109 | Commented-out weight preloading | Minor perf issue on multi-GPU | Low |







### 1.2 Missing Tensor Operations (CUDA Backend)







These panic at runtime when called:







| Operation | File | Impact |



|-----------|------|--------|



| `Add` | cuda_linux.go:189 | Fusion ops broken |



| `AddScalar` | cuda_linux.go:193 | Fusion ops broken |



| `Scale` | cuda_linux.go:197 | Fusion ops broken |



| `Tanh` | cuda_linux.go:213 | Activation support incomplete |



| `Slice` | cuda_linux.go:177 | Variable sequence handling broken |



| `Transpose view` | cuda_linux.go:181 | MatMul edge cases |







---







## 2. Feature Gaps vs Similar Tools







### 2.1 Quantization Support







| Feature | Status | Notes |



|---------|--------|-------|



| Q4_0 / Q8_0 (CPU) | ❌ Missing | No dequantization kernels |



| Q4_K / Q6_K (GPU) | ❌ Missing | Conversion script has TODO, runtime not implemented |



| GPTQ / AWQ | ❌ Missing | Not in scope yet |



| **Comparison**: sentence-transformers, llama.cpp | ✅ Has all | Critical gap |







**Recommendation**: Implement Q4/Q8 support for both Metal and CUDA to reduce VRAM requirements by 50-75%.







### 2.2 Model Ecosystem







| Model | Status |



|-------|--------|



| BERT-tiny | ✅ Done |



| Nomic-Embed-Text | ✅ Done |



| all-MiniLM-L6-v2 | ✅ Done |



| bge-m3 | ❌ Missing (roadmap) |



| e5-mistral | ❌ Missing (roadmap) |



| GTE | ❌ Missing |



| E5-v2 | ❌ Missing |



| **Comparison**: sentence-transformers | ✅ 100+ models |







**Recommendation**: Add 2-3 more popular embedding models to match baseline expectations.







### 2.3 Production Features







| Feature | Status | Priority |



|---------|--------|----------|



| mTLS for Flight/gRPC | ❌ Missing | High |



| API Key Auth | ❌ Missing | High |



| Dynamic model loading | ❌ Missing | Medium |



| Python SDK | ❌ Missing | High |



| Node.js client | ❌ Missing | Medium |



| OpenAPI/Swagger | ❌ Missing | Low |



| **Comparison**: Ollama | ✅ Has all | |







### 2.4 Observability







| Feature | Status | Priority |



|---------|--------|----------|



| OpenTelemetry | ✅ Basic | Done |



| Grafana dashboards | ⚠️ Templates missing | Medium |



| Structured logging (correlation IDs) | ❌ Missing | Low |



| pprof security | ❌ Missing | Low |







---







## 3. Recommended Next Steps







### Phase 1: Fix Critical TODOs (1-2 weeks)







#### Week 1: CUDA Backend Fixes







- [ ] **P0**: Implement `cudaMemGetInfo` in `cuda_linux.go:GetVRAMUsage()`



- [ ] **P0**: Implement `Cast()` kernel for FP16<->FP32 conversion



- [ ] **P1**: Implement missing tensor ops: `Add`, `AddScalar`, `Scale`, `Tanh`



- [ ] **P1**: Implement `Slice` for variable sequence lengths







#### Week 2: Performance Improvements







- [ ] **P1**: Add `Paste/SetSlice` to Tensor interface + implement for Metal/CUDA



- [ ] **P2**: Uncomment and fix weight preloading in embeddings.go







### Phase 2: Quantization (2-3 weeks)







- [ ] **P0**: Implement Q4_0/Q8_0 dequantization for Metal



- [ ] **P0**: Implement Q4_0/Q8_0 dequantization for CUDA



- [ ] **P1**: Add CLI flags `--quantization int8` / `int4`



- [ ] **P1**: Validate accuracy vs performance (target: <2% degradation)







### Phase 3: Model Expansion (2 weeks)







- [ ] **P1**: Add GTE (General Text Embeddings) support



- [ ] **P2**: Add E5-v2 support



- [ ] **P2**: Add bge-m3 support







### Phase 4: Production Hardening (2-3 weeks)







- [ ] **P0**: Implement mTLS for Arrow Flight



- [ ] **P1**: Create Node.js client library







### Phase 5: Observability (1 week)







- [ ] **P1**: Create Grafana dashboard templates



- [ ] **P2**: Add correlation IDs to structured logging



- [ ] **P2**: Add pprof endpoint security







---







## 4. Priority Matrix







```



                    | Low Effort | Medium Effort | High Effort |



|------------------|------------|---------------|--------------|



| High Impact      | VRAM metric| Cast kernels  | Quantization |



| Medium Impact   | Load metric| Paste/SetSlice| mTLS + Auth  |



| Low Impact      | Weight prep| Model add     | Python SDK   |



```







### Top 5 Immediate Actions







1. **Fix CUDA Cast()** - Currently crashes FP16 inference on Linux



2. **Implement Q4/Q8 quantization** - Major VRAM reduction (2x-4x)



3. **Add API Key Auth** - Required for production deployment



4. **Create Python SDK** - Primary integration point for users



5. **Add GTE model support** - Popular model, easy to add







---







## 5. Comparison with Competitors







| Feature | Fletcher | sentence-transformers | llama.cpp | Ollama |



|---------|----------|------------------------|-----------|--------|



| Metal GPU | ✅ | ✅ (MPS) | ✅ | ✅ |



| CUDA GPU | ⚠️ Broken | ✅ | ✅ | ✅ |



| CPU fallback | ✅ | ✅ | ✅ | ✅ |



| Quantization (Q4/Q8) | ❌ | ✅ | ✅ | ✅ |



| FP16 inference | ✅ | ✅ | ✅ | ✅ |



| Multi-model | ⚠️ 3 models | ✅ 100+ | ✅ 100+ | ✅ 100+ |



| mTLS/gRPC | ❌ | N/A | ✅ | ✅ |



| Python SDK | ❌ | ✅ (native) | ✅ | ✅ |



| HTTP server | ✅ | ✅ | ✅ | ✅ |



| Arrow Flight | ✅ | ❌ | ❌ | ❌ |







**Verdict**: Fletcher leads in Arrow Flight integration and pure Go implementation. Gaps are primarily quantization, model count, and Python SDK.







---







## 6. Files Requiring Changes







### Core Engine Changes







- `internal/device/cuda_linux.go` - Fix GetVRAMUsage, Cast, add missing ops



- `internal/device/metal_darwin.go` - Add Paste/SetSlice



- `internal/embeddings/model/bert.go` - Use Paste/SetSlice



- `internal/embeddings/metrics.go` - Uncomment load balance metric



- `internal/embeddings/embeddings.go` - Uncomment weight preloading







### New Files







- `internal/device/cuda_quantization.go` - Q4/Q8 kernels



- `internal/device/metal_quantization.go` - Q4/Q8 kernels



- `python/fletcher/` - Python SDK



- `cmd/fletcher/auth.go` - API Key middleware







---







## 7. Success Criteria







By completing Phase 1-3, Fletcher will have:







- ✅ Working CUDA FP16 inference



- ✅ 50-75% VRAM reduction via quantization



- ✅ 3 additional popular models



- ✅ Performance within 10% of sentence-transformers







By completing Phase 4-5, Fletcher will be production-ready:







- ✅ Secure transport (mTLS)



- ✅ Developer-friendly SDKs (Python, Node)



- ✅ Full observability stack







---







## 8. Mac Metal Performance Improvement Roadmap (Implementation Phase)







### 8.1 Current Performance Analysis







| Metric | Current | Target | Gap |



|--------|---------|--------|-----|



| Peak Throughput | ~24,200 vec/s | 30,000+ vec/s | +24% |



| Attention Kernel | Manual element-copy | Fused kernel | 10-15% |



| Batch Dispatch | Sequential | Pipelined | 5-10% |







### 8.2 Implementation Plan







#### Step 1: Model Preparation (Both Machines)







**Local Mac (Metal)**:



```bash



# Pull models via Ollama for reference testing



ollama pull nomic-embed-text



ollama pull bert-tiny



```







**Remote Linux (CUDA - ancalagon)**:



```bash



ssh ancalagon



cd ~/REPOS/longbow-fletcher



ollama pull nomic-embed-text



ollama pull bert-tiny



```







#### Step 2: CUDA Backend Fixes (ancalagon)







| Priority | Task | File | Line |



|----------|------|------|------|



| P0 | Implement `cudaMemGetInfo` | cuda_linux.go | 103 |



| P0 | Implement `Cast()` kernel | cuda_linux.go | 335 |



| P1 | Implement `Add` | cuda_linux.go | 189 |



| P1 | Implement `AddScalar` | cuda_linux.go | 193 |



| P1 | Implement `Scale` | cuda_linux.go | 197 |



| P1 | Implement `Tanh` | cuda_linux.go | 213 |



| P1 | Implement `Slice` | cuda_linux.go | 177 |







#### Step 3: Paste/SetSlice Implementation (Metal Performance)







| Task | File | Expected Improvement |



|------|------|---------------------|



| Add `Paste` to Tensor interface | internal/device/device.go | N/A |



| Implement for Metal | internal/device/metal_darwin.go | 10-15% |



| Implement for CPU | internal/device/cpu_backend.go | 5-10% |



| Use in bert.go attention | internal/embeddings/model/bert.go | Baseline |







#### Step 4: Testing Infrastructure







**Unit Tests**:



- `internal/device/cuda_linux_test.go` - Test all tensor operations



- `internal/device/metal_darwin_test.go` - Test Paste/SetSlice







**Fuzz Tests**:



- `internal/embeddings/fuzz_test.go` - Random text → embeddings → validate bounds







**Coherence Tests**:



- Compare Fletcher embeddings vs Ollama reference outputs



- Cosine similarity must be > 0.99 for same model



- Test with nomic-embed-text and bert-tiny







#### Step 5: Benchmark Suite







Create comparison scripts:



- `scripts/benchmark_metal_vs_ollama.py` - Mac Metal vs Ollama



- `scripts/benchmark_cuda_vs_ollama.py` - CUDA vs Ollama (ancalagon)



- `scripts/benchmark_cross_platform.py` - Metal vs CUDA coherence







### 8.3 Detailed Task Breakdown







```



Week 1: CUDA Fixes (ancalagon)



├── SSH to ancalagon



├── Fix GetVRAMUsage() 



├── Fix Cast() kernel



├── Implement Add/AddScalar/Scale/Tanh/Slice



├── Write unit tests for each operation



├── Build with -tags cuda



└── Run basic inference test







Week 2: Metal Performance (Local)



├── Add Paste to Tensor interface



├── Implement Metal Paste



├── Implement CPU Paste (fallback)



├── Update bert.go to use Paste



└── Run attention benchmark







Week 3: Testing & Validation



├── Pull models on both machines



├── Write fuzz tests



├── Write coherence tests vs Ollama



├── Verify embedding quality



└── Document results







Week 4: Benchmark & Compare



├── Run Metal benchmarks (local)



├── Run CUDA benchmarks (ancalagon)



├── Compare vs Ollama reference



├── Update nextsteps.md with results



└── Plan next optimizations



```







### 8.4 Test Requirements







**Unit Tests (Minimum)**:



- [ ] CUDA: Add, AddScalar, Scale, Tanh, Slice, Cast, GetVRAMUsage



- [ ] Metal: Paste/SetSlice correctness



- [ ] CPU: Paste fallback correctness



- [ ] BERT: End-to-end inference







**Fuzz Tests (Minimum)**:



- [ ] Random ASCII text → valid embeddings (no NaN/Inf)



- [ ] Random UTF-8 text → valid embeddings



- [ ] Empty/malformed input → graceful error







**Coherence Tests (Minimum)**:



- [ ] nomic-embed-text: cosine similarity vs Ollama > 0.99



- [ ] bert-tiny: cosine similarity vs Ollama > 0.99



- [ ] Cross-platform: Metal vs CUDA similarity > 0.99







### 8.5 Success Metrics







| Metric | Target | Measurement |



|--------|--------|-------------|



| CUDA inference | No panics | Run 1000 embeddings |



| Metal throughput | >28,000 vec/s | Benchmark with bert-tiny batch=256 |



| Embedding quality | >0.99 cosine vs Ollama | Coherence test |



| Fuzz test pass | 0 failures | 10,000 iterations |







---







## 9. SSH Access & Remote Execution







### ancalagon Connection







```bash



# Test SSH connectivity



ssh ancalagon "hostname"







# Navigate to repo



ssh ancalagon "cd ~/REPOS/longbow-fletcher && pwd"







# Build CUDA version



ssh ancalagon "cd ~/REPOS/longbow-fletcher && CGO_ENABLED=1 go build -tags cuda -o bin/fletcher ./cmd/fletcher"







# Run tests



ssh ancalagon "cd ~/REPOS/longbow-fletcher && CGO_ENABLED=1 go test -tags cuda ./internal/device/..."



```







---







## 10. File Changes Summary







### Modified Files







1. `internal/device/cuda_linux.go` - Fix all TODO/unimplemented items



2. `internal/device/device.go` - Add Paste method to interface



3. `internal/device/metal_darwin.go` - Implement Paste



4. `internal/device/cpu_backend.go` - Implement Paste fallback



5. `internal/embeddings/model/bert.go` - Use Paste in attention







### New Files







1. `internal/device/cuda_linux_test.go` - CUDA unit tests



2. `internal/device/cuda_fuzz_test.go` - CUDA fuzz tests



3. `internal/embeddings/fuzz_test.go` - Embedding fuzz tests



4. `internal/embeddings/coherence_test.go` - Ollama comparison tests



5. `scripts/benchmark_metal_vs_ollama.py` - Metal benchmark



6. `scripts/benchmark_cuda_vs_ollama.py` - CUDA benchmark







---







## 11. Implementation Checklist











---







## 12. Performance Comparison Findings (2026-03-24)







### Test Results Summary







| Machine | Backend | Model | Status | Notes |



|---------|---------|-------|--------|-------|



| Mac (M-series) | Metal | bert-tiny | ✅ Tests pass | Weights need downloading |



| Mac (M-series) | Metal | nomic-embed-text | ✅ Ollama ready | 274MB model |



| Ancalagon (Linux) | CPU | bert-tiny | ✅ Tests pass | 17MB weights |



| Ancalagon (Linux) | CPU | nomic-embed-text | ⚠️ Fails | Wrong weights format |



| Ancalagon (Linux) | CUDA | - | ⚠️ Stub only | memcpy, no real GPU ops |







### Key Issues Identified







1. **Model Weights**: nomic-embed-text requires conversion from HuggingFace safetensors



2. **CUDA Backend**: Current implementation uses simple memcpy stubs, not actual GPU kernels



3. **Weights Loading**: Test failures due to incorrect weights file format



4. **Ollama Integration**: Both machines have Ollama with nomic-embed-text for comparison







### Performance Baseline (CPU)







| Machine | bert-tiny throughput | Backend |



|---------|---------------------|---------|



| Mac | ~1,300-1,900 seq/s | Metal (CPU fallback) |



| Ancalagon (Linux) | ~2,200 seq/s | CPU (CUDA stubs) |







### Test Results (2026-03-24)







**Mac (Metal)**:



- TestEmbeddingCoherence_BertTiny: ✅ PASS (weights load)



- TestEmbeddingCoherence_NomicEmbed: ❌ SKIP (not BERT architecture)



- TestEmbeddingCoherence_SameTextSameEmbedding: ✅ PASS (1.0 similarity)



- TestEmbeddingCoherence_DifferentTextsDifferentEmbeddings: ✅ PASS (random weights)



- Fuzz tests: Some tests hang (needs investigation)







**Ancalagon (Linux/CUDA)**:



- TestEmbeddingCoherence_BertTiny: ✅ PASS (weights load)



- TestEmbeddingCoherence_NomicEmbed: ❌ SKIP (not BERT architecture)  



- TestEmbeddingCoherence_SameTextSameEmbedding: ✅ PASS (1.0 similarity)



- CUDA backend: Falls back to CPU (stubs not implemented)







---







## 13. Phase 2: Production Readiness Plan







### Part 1: Fix Model Weights Pipeline (Priority: Critical)







### Part 2: Implement Real CUDA GPU Kernels (Priority: Critical)







### Part 3: Metal Performance Optimization (Priority: High)



- [ ] Profile current Metal kernel bottlenecks







### Part 4: Ollama Comparison Tests (Priority: High)



  - Added `TestOllamaCoherence_BertTiny` and `TestOllamaCoherence_NomicEmbedText` in coherence_test.go



  - Uses Ollama REST API at http://localhost:11434/api/embeddings



  - Scripts: benchmark_quick.sh, benchmark_full.sh, benchmark_fair.sh



  - Results: Fletcher 16.5/s (CPU), Ollama 9.8/s (GPU)



  - Fletcher 1.6x faster despite using CPU vs Ollama's GPU



  - Different models: Fletcher=bert-tiny(128-dim), Ollama=nomic(768-dim)







### Part 5: Quantization Support (Priority: Medium)



  - PolarQuant: Cartesian to polar coordinate conversion



  - QJL: 1-bit Johnson-Lindenstrauss residual correction



  - 6x memory reduction with zero accuracy loss



  - Float32<->Float64, Int8/16/32/64, Uint8/16/32/64



  - Added dtype tracking to CudaTensor



  - Float32<->Float64, Int32/64, Uint32/64, Int8/Uint8



  - Added to Metal shaders and Objective-C++ backend



- [ ] Test with quantized nomic-embed-text







### Part 6: API & Auth (Priority: Medium)







### Part 7: Model Support Expansion (Priority: Medium)







### Part 8: Observability (Priority: Low)







### Part 9: Client SDKs (Priority: Low)







### Part 10: CI/CD & Testing (Priority: Low)



- [ ] Add GPU test runner to CI







---







## 14. vLLM Feature Parity Roadmap







Based on analysis of [vllm-project/vllm](https://github.com/vllm-project/vllm), this section outlines features needed for competitive feature parity.







### vLLM Key Features (What's Missing in Fletcher)







| Feature | vLLM Status | Fletcher Status | Priority |



|---------|-------------|------------------|----------|



| Multi-Modal (Images, Audio, Video) | ✅ Supported | ❌ Missing | P0 |



| Sparse Embeddings (SPLADE) | ✅ Supported | ❌ Missing | P1 |



| RoBERTa / XLM-RoBERTa | ✅ Supported | ❌ Missing | P1 |



| BERT variants | ✅ Supported | ⚠️ Basic | P1 |



| LoRA Support | ✅ Supported | ❌ Missing | P2 |



| Prefix Caching | ✅ Optimized | ❌ Missing | P2 |



| Chunked Prefill | ✅ Optimized | ❌ Missing | P2 |



| FP8 KV Cache | ✅ Supported (Hopper) | ❌ Missing | P2 |



| Spec Decode | ⚠️ WIP | ❌ N/A | P3 |



| Structured Output | ✅ Supported | ❌ Missing | P2 |



| Multiple Pooling Strategies | ✅ Supported | ⚠️ Basic | P1 |



| Model Registry/Auto-Discovery | ✅ Supported | ❌ Missing | P1 |



| HuggingFace Hub Integration | ✅ Full | ⚠️ Limited | P1 |



| OpenAI Compatible API | ✅ Full | ⚠️ Partial | P1 |



| AMD/TPU Support | ⚠️ WIP | ❌ Missing | P3 |







### Part 11: Multi-Modal Support (P0)



  - Added `VisionConfig`, `VisionModel`, `TransformerEncoder` in model/vision.go



  - Added `VisionEncoder` interface with ViT implementation



  - Added `MultiModalInput`, `EmbeddingInput` in embeddings/multimodal.go



  - Added `ImagePreprocessor` with CLIP normalization values



- [ ] Add image-to-text embedding pipeline



- [ ] Add multi-modal input preprocessing for HTTP API



- [ ] Support image embeddings via OpenAI-compatible API







### Part 12: Sparse Embeddings (P1)



  - Added `SparseConfig`, `SpladeEncoder`, `SparseEmbedding` in model/sparse.go



  - Supports max pooling with log-saturation activation



  - `SparseEmbedding` with map-based storage for efficient sparse vectors



  - `ToDense()` conversion for compatibility



  - Ready for hybrid search integration







### Part 13: Model Ecosystem Expansion (P1)







### Part 14: Advanced Features (P2)



  - Added Pooler with CLS, Mean, Max, Last strategies in model/pooler.go







### Part 15: OpenAI Compatibility (P1)







### Part 16: Hardware Support (P3)







---







### Feature Parity Priority Matrix







```



                          | Quick Win | Medium Effort | Long Term |



|------------------------|-----------|---------------|-----------|



| High Impact            | Ollama comp| Multi-modal  | Sparse emb|



|                        | Pooling   | Model expand | Full HF   |



| Medium Impact          | Prefix    | Structured   | LoRA      |



|                        | cache     | output       |           |



| Low Impact             | AMD ROCm  | TPU backend  | Spec dec  |



```







### Top 5 Priorities for vLLM Parity







1. **Multi-Modal Support** - Biggest gap vs vLLM



2. **Model Auto-Discovery** - Load any HF model automatically



3. **Full OpenAI Compatibility** - Drop-in replacement



4. **SPLADE/Sparse Embeddings** - Hybrid retrieval



5. **More Pooling Strategies** - Mean, CLS, Max, last-token







---







## 15. Multi-Dimension & Multi-Datatype Support







### Goals







Support all standard embedding dimensions and Go native datatypes for maximum flexibility.







### Supported Dimensions







| Dimension | Status | Use Case |



|-----------|--------|----------|



| 128 | ✅ Implemented | BERT-Tiny |



| 384 | ✅ Implemented | all-MiniLM-L6-v2 |



| 768 | ✅ Implemented | BERT-Base, Nomic |



| 1024 | ✅ Implemented | bge-m3 |



| 1536 | ✅ Implemented | Custom models |



| 2048 | ✅ Implemented | Large models |



| 3072 | ✅ Implemented | XL models |







### Supported Datatypes







| Category | Types | Status |



|----------|-------|--------|



| **Integers** | int, int8, int16, int32, int64 | ✅ Implemented |



| **Unsigned** | uint, uint8, uint16, uint32, uint64, uintptr | ✅ Implemented |



| **Float** | float32, float64 | ✅ Implemented |



| **Complex** | complex64, complex128 | ✅ Implemented |







### Implementation Plan







#### Task 1: Add Dimension Support (Priority: High)











#### Task 2: Add Datatype Support (Priority: High)











#### Task 3: Add Unit Tests (Priority: High)











#### Task 4: Add Fuzz Tests (Priority: High)











### Files to Modify







1. `internal/device/device.go` - Add datatype constants and interface methods



2. `internal/device/cpu_backend.go` - Add datatype-specific implementations



3. `internal/device/metal_darwin.go` - Add Metal datatype kernels



4. `internal/device/cuda_linux.go` - Add CUDA datatype kernels



5. `internal/embeddings/model/bert.go` - Add dimension configs



6. `internal/embeddings/embeddings.go` - Add dimension validation



7. `internal/device/*_test.go` - Add unit tests



8. `internal/embeddings/*_test.go` - Add fuzz tests







### Success Criteria

## 9. Batched Implementation Plan (Subtasks)

### Batch A - Protocol and Transport (High Impact)
- [ ] A.2 Implement Longbow namespace path parsing in client (`namespace/dataset`).
- [ ] A.7 Add CI job building against Longbow module versions.

### Batch B - Security and Observability (High Impact)
- [ ] B.2 Add tenant propagation to all client requests.
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
## 16. God Module Refactoring (Context Window Optimization)

Large "god" modules should be split into smaller, focused files/packages so they fit
more easily into context windows and are easier to maintain.

- [ ] 16.1 Identify modules over 1000 lines (`embeddings.go`, `server.go`, `bert.go`, `device.go`).
- [ ] 16.2 Split `embeddings.go` into tokenizer, batching, caching, and metrics components.
- [ ] 16.3 Split `server.go` into HTTP handlers, middleware, and OpenAPI handlers.
- [ ] 16.4 Split `bert.go` into layers, attention, and pooling submodules.
- [ ] 16.5 Split device backends into per-operation files.
- [ ] 16.6 Add package-level documentation for each new module.
- [ ] 16.7 Update imports and tests after refactor.