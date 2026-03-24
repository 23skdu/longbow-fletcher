# Longbow Fletcher: Deep Analysis & Next Steps

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
- [ ] **P0**: Add API Key authentication middleware
- [ ] **P1**: Create Python SDK (`pip install fletcher`)
- [ ] **P1**: Create Node.js client library
- [ ] **P2**: Publish OpenAPI spec

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

- [x] SSH to ancalagon and verify access
- [x] Pull nomic-embed-text on ancalagon
- [x] Pull bert-tiny on ancalagon
- [x] Pull nomic-embed-text locally
- [x] Pull bert-tiny locally
- [x] Fix CUDA GetVRAMUsage()
- [x] Fix CUDA Cast()
- [x] Implement CUDA Add/AddScalar/Scale/Tanh/Slice
- [x] Write CUDA unit tests
- [x] Test CUDA inference
- [x] Add Paste to Tensor interface
- [x] Implement Metal Paste
- [x] Update bert.go to use Paste
- [x] Write fuzz tests
- [x] Write coherence tests
- [x] Run Metal benchmark
- [x] Run CUDA benchmark
- [x] Compare results and document
