# Fletcher Performance Notes

Fletcher targets maximum throughput on commodity hardware. Use `cmd/bench` and
`scripts/run_benchmark.py` to measure local performance and detect regressions.

## Measured Metrics

- **Vectors/sec**: end-to-end embedding throughput.
- **Latency**: mean, P50, P95, P99 per batch.
- **Memory**: heap allocation and object counts.
- **Concurrency**: goroutine count during run.

## Recommended Workflow

1. Build: `make build-bench`
2. Baseline: `python3 scripts/run_benchmark.py --runs 10`
3. Store `benchmark_results/summary.json` as baseline.
4. After changes, rerun and compare throughput/latency.
5. Investigate regressions > 5%.

## Backend Notes

- **CPU**: BLAS-accelerated (OpenBLAS/Accelerate).
- **Metal**: Apple Silicon FP16 kernels.
- **CUDA**: cuBLAS + custom kernels.

## Tips

- Use consistent batch sizes and text lengths across runs.
- Warmup is enabled by default.
- Run on an idle machine for stable numbers.