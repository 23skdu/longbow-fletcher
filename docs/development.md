# Development Guide

## Testing

Fletcher is designed to be a first-class testing tool for Longbow vector stores, capable of generating stress test loads.

### Stress Testing (Lorem Ipsum)

Generate and ingest random "Lorem Ipsum" traffic to test cluster capacity:

```bash
./bin/fletcher --vocab vocab.txt --lorem 100 --server localhost:3000 --dataset stress_test
```

### Integration Testing with Ops Scripts

1. **Start Longbow Cluster**: via `start_local_cluster.sh`.
2. **Populate Data**: Run Fletcher to ingest 50 items.
3. **Verify**: Use `longbow/scripts/ops_test.py` to query and validate.

## Benchmarks

**Hardware**: Apple M3 Pro (12 Cores, 18GB RAM)  
**Model**: BERT-Tiny (FP16/Metal vs FP16/MPS)

| Scenario | Fletcher (Metal) | PyTorch (MPS) | Result |
| :--- | :--- | :--- | :--- |
| **Peak Throughput** | **~24,200 vec/s** | ~14,800 vec/s | **1.6x Faster** |
| **Sustained (500K)** | **~21,000 vec/s** | ~8,300 vec/s | **2.5x Faster** |
| **Memory Footprint** | **< 10 MB** | ~370 MB | **37x Lighter** |

**Advantages**:

- Zero-overhead CGO dispatch to Metal kernels.
- Buffer pooling prevents allocation churn.
- No thermal throttling drops commonly seen in Python/PyTorch loops.

## Client Scripts

Python scripts are provided in `scripts/` for interacting with the Fletcher server.

### Prerequisites

```bash
pip install cbor2 httpx pyarrow numpy
```

### Grpc Client (`grpc_client_example.py`)

Sends text to Fletcher via HTTP/2 + CBOR.

```bash
python scripts/grpc_client_example.py -s http://localhost:8080 "Hello World"
```

### Arrow Flight Client (`arrow_flight_client.py`)

Performs distributed vector search against Longbow via Arrow Flight.

```bash
# Vector Search
python arrow_flight_client.py search --dataset mydata --text "query" --k 10

# Distributed Search (Global)
python arrow_flight_client.py search --dataset mydata --global
```

**Key Flags**:

- `--global`: Enable cross-node distributed search.
- `--routing-key`: Route to specific shard.
