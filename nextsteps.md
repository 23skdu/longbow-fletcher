# Fletcher Optimization Roadmap: Phase 3

**Focus**: Performance Maximization & Production Stability
**Status**: Numerical Refinement Complete (1.0000 Sim)

## High Priority: Performance & Scalability (Weeks 5-6)

1. **[Performance] Implement SIMD Tokenization**
    - Replace generic string splitting with NEON/AVX2 optimized WordPiece tokenizer to remove CPU bottlenecks.

2. **[Performance] Implement Flash Attention (Metal)**
    - Implement IO-aware attention kernel to reduce VRAM BW usage by 3-4x for sequences >128.

3. **[Performance] Multi-Stream GPU Dispatch**
    - usage concurrent command encoders to overlap copy, compute, and host-transfer operations.

4. **[Performance] SwiGLU Fused Kernel Optimization**
    - Further optimize the Nomic SwiGLU kernel to use SIMD groups and reduce register pressure.

5. **[Performance] RoPE Fused Kernel Optimization**
    - Optimize Rotary Positional Embeddings to minimize complex number arithmetic overhead.

6. **[Performance] Async Flight Streaming**
    - Implement `DoPut` stream without blocking on batch completion to saturate network link.

7. **[Performance] Embedding Compression (FP16 Transport)**
    - Implement zero-copy FP16 Arrow transport to halve network bandwidth usage.

8. **[Performance] Dynamic Batch Sizing**
    - Implement adaptive batching size based on current sequence length distribution and VRAM availability.

9. **[Performance] Memory Arena Allocator**
    - Replace `sync.Pool` with a slab/arena allocator for tensors to guarantee locality and reduce fragmentation.

10. **[Performance] Dataset-Aware Caching**
    - Implement an LRU cache for frequently requested text hashes to bypass inference entirely.

## High Priority: Stability & Reliability (Weeks 7-8)

1. **[Stability] Graceful Backend Fallback**
    - Automatically fall back to CPU backend if Metal GPU fails (OOM, hang) without dropping requests.

2. **[Stability] VRAM Admission Control v2**
    - Replace heuristic VRAM estimation with real-time `query_resource_usage` Metal API feedback.

3. **[Stability] Output Validation (NaN Guard)**
    - Add low-overhead kernel to check for NaNs in final embeddings and reject/warn before sending.

4. **[Stability] Circuit Breaker Pattern**
    - Implement circuit breakers for Longbow connection failures to prevent cascade waiting.

5. **[Stability] Request Timeouts & Cancellation**
    - Propagate context cancellations down to the GPU command buffer (via `commit` check) to stop wasted work.

6. **[Observability] Prometheus Granular Metrics**
    - Add per-layer latency histograms and GPU utilization heatmaps.

7. **[Observability] Distributed Tracing (OpenTelemetry)**
    - Instrument the full request lifecycle (HTTP -> Tokenizer -> GPU -> Flight) with trace context.

8. **[Stability] Health Check Probes**
    - Implement `/healthz` and `/readyz` endpoints that perform actual dummy inference checks.

9. **[Infrastructure] End-to-End CI Benchmark Suite**
    - Automated regression testing for throughput (vecs/s) and latency (p99) on every commit.

10. **[Stability] Panic Recovery & Isolation**
    - Ensure individual request panics (e.g., malformed input) are recovered in the worker pool without crashing the process.
