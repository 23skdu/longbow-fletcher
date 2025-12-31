# Fletcher Client Scripts

Python client scripts for interacting with Fletcher embedding server and Longbow vector store.

## Prerequisites

```bash
# For grpc_client_example.py (HTTP/2 + CBOR)
pip install cbor2 httpx

# For arrow_flight_client.py (Arrow Flight)
pip install pyarrow numpy
```

---

## grpc_client_example.py

Sends text to the Fletcher embedding server and returns embedding vectors via HTTP/2 + CBOR.

### Usage

```bash
# Single text
python scripts/grpc_client_example.py -s http://localhost:8080 "Hello, world!"

# Multiple texts
python scripts/grpc_client_example.py -s http://localhost:8080 "Text one" "Text two"

# Via stdin pipe
echo "Hello from pipe" | python scripts/grpc_client_example.py

# File input
cat texts.txt | python scripts/grpc_client_example.py

# Target specific dataset for indexing
python scripts/grpc_client_example.py -D mydataset "Hello world"

# JSON output
python scripts/grpc_client_example.py --json "Hello world"
```

### Options

| Option | Description |
|--------|-------------|
| `--server, -s` | Fletcher server URL (default: `http://localhost:8080`) |
| `--dataset, -D` | Target dataset name for indexing in Longbow |
| `--timeout, -t` | Request timeout in seconds (default: 30) |
| `--json, -j` | Output embeddings as JSON |
| `--dims` | Print embedding dimensions only |

---

## arrow_flight_client.py

Arrow Flight client for Longbow vector search operations with support for distributed search.

### Usage

```bash
# Vector search with random query
python scripts/arrow_flight_client.py search --dataset mydata --dim 128 --k 10

# Search with explicit vector
python scripts/arrow_flight_client.py search --dataset mydata --vector "[0.1, 0.2, ...]" --k 5

# Distributed search with routing key
python scripts/arrow_flight_client.py search --dataset mydata --routing-key "shard-1"

# Global distributed search (all nodes)
python scripts/arrow_flight_client.py search --dataset mydata --global

# Hybrid search (vector + text)
python scripts/arrow_flight_client.py search --dataset mydata --text "query text" --alpha 0.5

# List all datasets
python scripts/arrow_flight_client.py list

# Get dataset info
python scripts/arrow_flight_client.py info --dataset mydata

# Cluster status
python scripts/arrow_flight_client.py status
```

### Commands

| Command | Description |
|---------|-------------|
| `search` | Vector similarity search |
| `list` | List all datasets |
| `info` | Get dataset information |
| `status` | Get cluster status |

### Global Options

| Option | Description |
|--------|-------------|
| `--data-uri, -d` | Longbow data server URI (default: `grpc://localhost:3000`) |
| `--meta-uri, -m` | Longbow meta server URI (default: `grpc://localhost:3001`) |
| `--json, -j` | Output as JSON |
| `--routing-key, -r` | Explicit routing key (`x-longbow-key` header) |

### Search Options

| Option | Description |
|--------|-------------|
| `--dataset` | Dataset name (required) |
| `--dim` | Vector dimension for random query (default: 128) |
| `--k` | Top K results (default: 10) |
| `--vector` | Query vector as JSON array |
| `--text` | Text query for hybrid search |
| `--alpha` | Hybrid alpha: 0=sparse, 1=dense (default: 0.5) |
| `--global` | Enable global distributed search |

---

## Distributed Search Headers

Both clients support Longbow's distributed search headers:

| Header | Description |
|--------|-------------|
| `x-longbow-key` | Route request to specific shard by key |
| `x-longbow-global` | Enable cross-node distributed search |

Example: Search across all cluster nodes:

```bash
python arrow_flight_client.py search --dataset mydata --global
```
