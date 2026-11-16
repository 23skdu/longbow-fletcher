#!/usr/bin/env python3
"""Orchestrate Fletcher Go benchmark runs and produce JSON + Markdown reports."""

import argparse
import json
import os
import subprocess
import statistics
from datetime import datetime, timezone
from pathlib import Path


def run_once(binary, args):
    cmd = [binary] + args
    out = subprocess.check_output(cmd, text=True)
    # bench prints JSON to stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./bin/bench")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--model", default="bert-tiny")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--output-dir", default="benchmark_results")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_args = [
        f"--model={args.model}",
        f"--precision={args.precision}",
        f"--batch={args.batch}",
        f"--iters={args.iters}",
    ]
    if args.gpu:
        run_args.append("--gpu")

    results = []
    for i in range(args.runs):
        r = run_once(args.binary, run_args + [f"--output={out_dir / f'run_{i}.json'}"])
        results.append(r)
        print(f"run {i+1}/{args.runs}: {r['vectors_per_second']:.1f} vec/s")

    vps = [r["vectors_per_second"] for r in results]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "precision": args.precision,
        "gpu": args.gpu,
        "batch": args.batch,
        "iters": args.iters,
        "runs": args.runs,
        "vectors_per_second": {
            "mean": statistics.mean(vps),
            "stdev": statistics.pstdev(vps),
            "min": min(vps),
            "max": max(vps),
        },
        "mean_latency_ms": statistics.mean([r["mean_latency_ms"] for r in results]),
        "p99_latency_ms": statistics.mean([r["p99_latency_ms"] for r in results]),
        "raw": results,
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    md_path = out_dir / "report.md"
    md_path.write_text(
        f"# Fletcher Benchmark Report\n\n"
        f"- Timestamp: {summary['timestamp']}\n"
        f"- Model: {args.model}\n"
        f"- Precision: {args.precision}\n"
        f"- GPU: {args.gpu}\n"
        f"- Batch: {args.batch}\n"
        f"- Iterations: {args.iters}\n"
        f"- Runs: {args.runs}\n\n"
        f"## Throughput\n\n"
        f"- Mean: {summary['vectors_per_second']['mean']:.1f} vec/s\n"
        f"- StdDev: {summary['vectors_per_second']['stdev']:.1f}\n"
        f"- Min: {summary['vectors_per_second']['min']:.1f}\n"
        f"- Max: {summary['vectors_per_second']['max']:.1f}\n\n"
        f"## Latency\n\n"
        f"- Mean: {summary['mean_latency_ms']:.3f} ms\n"
        f"- P99: {summary['p99_latency_ms']:.3f} ms\n"
    )
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
