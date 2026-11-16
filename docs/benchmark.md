# Benchmarking Fletcher

Fletcher includes a pure-Go benchmark tool (`cmd/bench`) and a Python orchestrator
(`scripts/run_benchmark.py`) for local performance and regression testing.

## Go Benchmark Tool

Build:

```bash
make build-bench
```

Run:

```bash
./bin/bench --model bert-tiny --batch 32 --iters 50 --output result.json
```

Key flags:

- `--model`: model type (`bert-tiny`, `nomic-embed-text`, ...)
- `--precision`: `fp32` or `fp16`
- `--gpu`: enable GPU backend
- `--batch`: batch size
- `--iters`: iterations
- `--warmup`: warmup iterations
- `--text-len`: words per generated text
- `--output`: write JSON result to file

Output JSON includes throughput (vec/s), latency percentiles, heap stats, and
goroutine count.

## Python Orchestrator

Run multiple iterations and produce a Markdown report + JSON summary:

```bash
python3 scripts/run_benchmark.py --binary ./bin/bench --runs 5 --batch 32 --iters 50
```

Outputs are written to `benchmark_results/`:

- `run_N.json` – per-run results
- `summary.json` – aggregated statistics
- `report.md` – human-readable report

Compare `summary.json` across runs/commits for regression detection.