# Testing Longbow with Fletcher

Fletcher is designed to be a first-class testing tool for Longbow vector stores. It can generate embeddings and push them directly to a running cluster.

## Prerequisites

1. A running Longbow server or cluster (see `longbow/scripts/start_local_cluster.sh`).
2. A vocabulary file (e.g., `vocab.txt`).

## Stress Testing with Lorem Ipsum

Generate and send a large number of embeddings to test ingest performance:

```bash
# Send 100 paragraphs to the local cluster
./bin/fletcher --vocab vocab.txt --lorem 100 --server localhost:3000 --dataset stress_test
```

## Integration with `ops_test.py`

1. Start the Longbow cluster.
2. Run Fletcher to populate a dataset.
3. Use `ops_test.py` to query and validate results.

```bash
# Terminal 1: Start Longbow
cd /path/to/longbow && ./scripts/start_local_cluster.sh

# Terminal 2: Run Fletcher
./bin/fletcher --vocab vocab.txt --lorem 50 --server localhost:3000 --dataset test_data

# Terminal 3: Run ops_test.py
cd /path/to/longbow && python scripts/ops_test.py --dataset test_data
```

## Docker-based Testing

For isolated testing, use Docker Compose or run Fletcher in a container:

```bash
docker run --rm --network host longbow-fletcher \
  --vocab /app/vocab.txt --lorem 25 --server localhost:3000
```
