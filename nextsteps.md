# Fletcher Optimization Roadmap

**Longbow-Fletcher**: High-Performance Text Embedding Engine for Longbow Vector Database

## Architecture Overview

**Core Components**:

- **Embedder**: Multi-GPU dispatch with load balancing, parallel tokenization, streaming results
- **BertModel**: Transformer architecture (BERT-tiny, Nomic-Embed-Text) with RoPE, SwiGLU, LayerNorm
- **Device Abstraction**: Metal (Apple Silicon), CUDA (NVIDIA), CPU (BLAS) backends
- **Flight Integration**: Apache Arrow Flight protocol for zero-copy Longbow communication
- **Servers**: HTTP REST API + Flight gRPC server with admission control

**Current Performance**:

- Metal GPU: ~24,200 vec/s peak, ~21,000 vec/s sustained
- 1.6x faster than PyTorch MPS, 2.5x sustained, 9.9x lower latency
- Batch size: 256 (Metal), 512 (CUDA), 32 (CPU)

---

## Priority 1: Numerical Refinement & Accuracy (Current Focus)

### Goal: Improve Cosine Similarity from 0.63 to >0.99 (vs HuggingFace)

- [x] **Step 1: Verify Tokenizer** (Vocab & ID generation)
  - Verified vocab match and punctuation splitting logic.
- [x] **Step 2: Verify Weights Loading** (Layout & Checksums)
  - Verified binary weights match PyTorch reference (confirmed strict adherence to transposition).

### Step 3: Verify Embeddings Layer (Word+Pos+Type+LN) 🔍 NEXT

- [ ] Isolate Word Embedding lookup vs PyTorch
- [ ] Isolate Position Embedding lookup vs PyTorch
- [ ] Isolate Token Type Embedding lookup vs PyTorch
- [ ] Verify LayerNorm (Gamma/Beta) accumulation
- [ ] Verify final Embeddings output tensor

### Step 4: Verify Attention Projections (Q/K/V)

- [ ] Inspect Query projection output
- [ ] Inspect Key projection output
- [ ] Inspect Value projection output
- [ ] Check for axis permutation in projection results

### Step 5: Verify Attention Mechanism (Scores & Softmax) 🚩 CRITICAL

- [ ] Inspect raw scores (Q * K^T)
- [ ] Inspect scaled scores (/ sqrt(d))
- [ ] Inspect Softmax probabilities (Sum=1.0)
- [ ] Check mask application (if any)

### Step 6: Verify Attention Context & Output

- [ ] Inspect context layer (Scores * V) - verify weighted sum logic
- [ ] Inspect Self-Attention Output projection (Dense + Residual + LN)

### Step 7: Verify Intermediate & Layer Outputs

- [ ] Inspect Intermediate Dense (Up projection)
- [ ] Inspect Activation function (GELU accuracy)
- [ ] Inspect Output Dense (Down projection) + Residual + LN

### Step 8: Verify Pooler Output

- [ ] Inspect Pooler Dense projection
- [ ] Inspect Tanh activation
- [ ] Check [CLS] extraction logic

### Step 9: Verify Metal Backend Consistency

- [ ] Run identical inputs on CPU vs Metal
- [ ] Verify FP16 precision tolerance

### Step 10: End-to-End Verification

- [ ] Achieve >0.99 cosine similarity on `verify_outputs.py`
- [ ] Validate Nomic specific features (RoPE, SwiGLU) with reference

---

## Priority 2: Performance & Throughput

### Step 1: Metal Kernel Optimization ⚡ HIGH IMPACT ✅ COMPLETE

**Completed**: Profiling infrastructure, baseline metrics, fused attention kernel

- [x] Profile Metal kernels to identify bottlenecks (MatMul, LayerNorm, Attention)
  - **Result**: MatMul 425 GFLOPS (optimal via MPS), Attention 13 GFLOPS (bottleneck)
- [x] Benchmark kernel-level performance vs theoretical peak FLOPS
  - **Result**: MatMul at 12% of peak (typical for memory-bound), LayerNorm 200x faster than threshold
- [x] Implement fused attention kernel (scale fusion into MatMul alpha)
  - **Result**: 3 dispatches vs 4, equivalent performance (dispatch overhead minimal ~3%)
- [ ] Implement tiled matrix multiplication for better cache utilization
  - **Skipped**: MPS already provides optimal tiling
- [ ] Optimize threadgroup memory usage in attention kernels
  - **Future**: Flash Attention for 2-3x improvement on long sequences
- [ ] Add Metal Performance Shaders (MPS) Graph integration for automatic optimization
  - **Partial**: Already using MPS for MatMul, could expand to other ops
- [ ] Implement FP16 compute with FP32 accumulation for precision/speed balance
  - **Implemented**: FP16 backend with FP32 accumulation in softmax

**Key Learning**: MPS is already optimal for standard operations. Focus on algorithmic improvements (Flash Attention) rather than micro-optimizations.

**Files**: 647 lines across 6 files  
**Commits**: `3f11844`, `2813329`, `f0febfd`

---

### Step 2: Multi-GPU Load Balancing Improvements ✅ COMPLETE

**Completed**: Dynamic weighted distribution, performance metrics, Prometheus monitoring

- [x] Implement dynamic load balancing based on actual GPU utilization
  - **Result**: Weighted distribution using throughput metrics (sequences/second)
- [x] Profile token-based vs sequence-based load distribution
  - **Result**: Token-based with dynamic weights adapts to GPU performance
- [x] Add GPU memory pressure monitoring for adaptive batching
  - **Result**: VRAM usage metrics exposed via Prometheus
- [x] Implement cross-GPU result aggregation optimization
  - **Result**: Streaming results via channels, no blocking
- [ ] Add GPU affinity for better NUMA performance
  - **Future**: Requires multi-socket testing
- [ ] Implement work-stealing between GPUs for unbalanced workloads
  - **Future**: Phase 3 optimization

**Performance**: 10-20% improvement for imbalanced workloads, 0% overhead for balanced

**Prometheus Metrics**:

- `fletcher_gpu_throughput{device}` - Sequences/sec per GPU
- `fletcher_gpu_batch_time_seconds{device}` - Batch processing time
- `fletcher_gpu_weight{device}` - Load balancing weight
- `fletcher_gpu_sequences_total{device}` - Total sequences processed
- `fletcher_gpu_tokens_total{device}` - Total tokens processed

**Files**: 144 lines across 2 files  
**Commits**: `e77b7ae`, `f09c508`

---

### Step 3: Tokenization Parallelization ✅ COMPLETE

**Completed**: Work queue distribution, pre-allocated buffers, tokenization metrics

- [x] Benchmark current parallel tokenization overhead
  - **Result**: ~15-20% of total time, dominated by string operations
- [x] Optimize memory allocation in tokenization workers
  - **Result**: Pre-allocated buffers (1.5 tokens/word), 20-30% fewer allocations
- [x] Implement work queue for better load balancing
  - **Result**: Dynamic work stealing, better for imbalanced text lengths
- [ ] Implement SIMD-accelerated WordPiece tokenization
  - **Future**: Requires custom string operations
- [ ] Add tokenization result caching for repeated texts
  - **Future**: Requires cache management
- [ ] Implement streaming tokenization for large texts
  - **Future**: For documents >10K tokens
- [ ] Add vocabulary trie optimization for faster lookups
  - **Future**: Requires trie data structure

**Performance**: 5-10% end-to-end improvement for imbalanced workloads

**Prometheus Metrics**:

- `fletcher_tokenization_duration_seconds` - Tokenization time
- `fletcher_tokenization_throughput` - Tokens/second

**Files**: 51 lines across 3 files  
**Commit**: `aa77919`

---

### Step 4: Zero-Copy Data Paths

- [ ] Audit all memory copies in hot path (tokenization → inference → output)
- [ ] Implement memory-mapped weight loading
- [ ] Add zero-copy Arrow RecordBatch construction
- [ ] Optimize FP16 transport format (avoid FP32 conversion)
- [ ] Implement GPU-direct RDMA for multi-node deployments
- [ ] Profile memory bandwidth utilization

### Step 5: Batching & Admission Control

- [ ] Implement dynamic batch sizing based on sequence lengths
- [ ] Add request coalescing for concurrent requests
- [ ] Optimize VRAM estimation algorithm (currently heuristic-based)
- [ ] Implement priority queuing for latency-sensitive requests
- [ ] Add adaptive timeout based on queue depth
- [ ] Profile admission control overhead

---

## Priority 2: Longbow Integration

### Step 6: Flight Protocol Optimization

- [ ] Implement streaming DoPut for large batches (avoid buffering)
- [ ] Add compression for Flight transfers (LZ4/Zstd)
- [ ] Optimize Arrow schema for minimal overhead
- [ ] Implement connection pooling for Flight clients
- [ ] Add retry logic with exponential backoff
- [ ] Profile Flight vs HTTP performance

### Step 7: Embedding Format Optimization

- [ ] Add configurable embedding precision (FP32/FP16/INT8)
- [ ] Implement embedding quantization for storage
- [ ] Add embedding normalization options (L2, unit sphere)
- [ ] Optimize Arrow FixedSizeList layout
- [ ] Implement sparse embedding support
- [ ] Add embedding metadata (model version, timestamp)

### Step 8: Batch Streaming to Longbow

- [ ] Implement incremental batch sends (don't wait for full batch)
- [ ] Add backpressure handling from Longbow
- [ ] Optimize batch size for network MTU
- [ ] Implement parallel Flight streams for multi-GPU
- [ ] Add progress reporting for large uploads
- [ ] Profile end-to-end latency (Fletcher → Longbow)

### Step 9: Distributed Fletcher Deployment

- [ ] Implement Fletcher cluster coordination
- [ ] Add load balancing across Fletcher instances
- [ ] Implement shared model weight storage (S3/NFS)
- [ ] Add health checks and auto-recovery
- [ ] Implement request routing based on model type
- [ ] Add distributed tracing across Fletcher cluster

### Step 10: Longbow-Aware Optimizations

- [ ] Implement dataset-specific embedding caching
- [ ] Add incremental embedding updates (delta encoding)
- [ ] Optimize for Longbow's HNSW index requirements
- [ ] Implement embedding deduplication before sending
- [ ] Add Longbow capacity monitoring
- [ ] Profile Longbow ingestion bottlenecks

---

## Priority 3: Model Support & Quality

### Step 11: Additional Model Architectures

- [ ] Add Sentence-BERT support
- [ ] Implement E5 embedding models
- [ ] Add multilingual model support (XLM-R)
- [ ] Implement domain-specific models (code, biomedical)
- [ ] Add model hot-swapping without downtime
- [ ] Implement model versioning and A/B testing

### Step 12: Advanced Transformer Features

- [ ] Implement Flash Attention for long sequences
- [ ] Add ALiBi positional encoding support
- [ ] Implement sparse attention patterns
- [ ] Add cross-attention for multi-modal embeddings
- [ ] Implement model distillation for smaller models
- [ ] Add quantization-aware training support

### Step 13: Dynamic Sequence Length Handling

- [ ] Implement adaptive padding/truncation
- [ ] Add sequence bucketing for efficient batching
- [ ] Optimize for variable-length sequences
- [ ] Implement sliding window for long documents
- [ ] Add hierarchical embedding (sentence → document)
- [ ] Profile memory usage vs sequence length

### Step 14: Embedding Quality Improvements

- [ ] Add embedding quality metrics (cosine similarity distribution)
- [ ] Implement embedding calibration
- [ ] Add contrastive learning fine-tuning
- [ ] Implement hard negative mining
- [ ] Add embedding dimensionality reduction (PCA/UMAP)
- [ ] Profile embedding separability

### Step 15: Model Quantization

- [ ] Implement INT8 quantization for weights
- [ ] Add dynamic quantization for activations
- [ ] Implement mixed-precision inference (FP16 + INT8)
- [ ] Add quantization-aware fine-tuning
- [ ] Profile accuracy vs speed tradeoffs
- [ ] Implement quantization for Metal/CUDA backends

---

## Priority 4: Production Readiness

### Step 16: Observability & Monitoring

- [ ] Add detailed per-request tracing (OpenTelemetry)
- [ ] Implement Prometheus metrics for all operations
- [ ] Add GPU utilization monitoring
- [ ] Implement request latency percentiles (p50, p95, p99)
- [ ] Add error rate tracking by error type
- [ ] Implement distributed tracing with Longbow

### Step 17: Reliability & Error Handling

- [ ] Add graceful degradation when GPU unavailable
- [ ] Implement automatic fallback to CPU backend
- [ ] Add circuit breaker for failing models
- [ ] Implement request timeout and cancellation
- [ ] Add retry logic for transient failures
- [ ] Implement health checks for all components

### Step 18: Resource Management

- [ ] Implement GPU memory defragmentation
- [ ] Add automatic garbage collection for unused tensors
- [ ] Optimize tensor pooling (reduce allocations)
- [ ] Implement memory pressure detection
- [ ] Add CPU/GPU affinity optimization
- [ ] Profile memory fragmentation over time

### Step 19: Security & Compliance

- [ ] Add authentication for HTTP/Flight servers
- [ ] Implement rate limiting per client
- [ ] Add input validation and sanitization
- [ ] Implement audit logging for all requests
- [ ] Add encryption for Flight transfers (TLS)
- [ ] Implement data retention policies

### Step 20: Developer Experience

- [ ] Add comprehensive API documentation (OpenAPI)
- [ ] Implement client SDKs (Python, Go, Rust)
- [ ] Add interactive playground/demo
- [ ] Create benchmarking suite
- [ ] Add model conversion tools (PyTorch → Fletcher)
- [ ] Implement debugging tools (tensor inspection, profiling)

---

## Implementation Priority

**Phase 1 (Weeks 1-2)**: Steps 1-5 (Performance & Throughput)  
**Phase 2 (Weeks 3-4)**: Steps 6-10 (Longbow Integration)  
**Phase 3 (Weeks 5-6)**: Steps 11-15 (Model Support & Quality)  
**Phase 4 (Weeks 7-8)**: Steps 16-20 (Production Readiness)

## Success Metrics

- **Throughput**: >30,000 vec/s sustained on M3 Pro
- **Latency**: <0.3ms single-sequence latency
- **Efficiency**: >80% GPU utilization during sustained load
- **Reliability**: 99.9% uptime, <0.1% error rate
- **Integration**: <10ms end-to-end latency to Longbow

## Current Bottlenecks

Based on codebase analysis:

1. **Metal Kernels**: MatMul not fully optimized (tiling, threadgroup memory)
2. **Load Balancing**: Token-based balancing is heuristic, not adaptive
3. **Memory Copies**: Multiple FP32↔FP16 conversions in output path
4. **Admission Control**: VRAM estimation is conservative, underutilizes GPU
5. **Flight Integration**: Single-threaded DoPut, no streaming

## Longbow-Specific Optimizations

**Current Integration**:

- Apache Arrow Flight for data transfer
- Schema: `{text: string, embedding: fixed_size_list<float32>[dim]}`
- Batch-based transfers

**Optimization Opportunities**:

1. **Streaming Embeddings**: Don't wait for full batch completion
2. **Embedding Compression**: FP16 transport saves 50% bandwidth
3. **Deduplication**: Hash-based dedup before sending to Longbow
4. **Parallel Streams**: One Flight stream per GPU
5. **Backpressure**: Monitor Longbow ingestion rate, adapt batch size
6. **Metadata**: Add model version, timestamp for Longbow indexing

---

## Notes

- Fletcher is optimized for throughput, not latency (batching-first design)
- Metal backend is most mature (24K+ vec/s), CUDA needs optimization
- Current bottleneck is likely Metal kernel efficiency, not tokenization
- Longbow integration is basic (DoPut only), needs streaming + compression
- Production deployment needs better observability and error handling
- Model support is limited (BERT-tiny, Nomic), needs expansion
