# Fletcher Scripts

Python client scripts for Fletcher and Longbow.

## Quick Start

```bash
# Install dependencies
pip install cbor2 httpx pyarrow numpy

# Embed text via Fletcher
python grpc_client_example.py -s http://localhost:8080 "Hello world"

# Vector search via Longbow
python arrow_flight_client.py search --dataset mydata --dim 128 --k 10
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `grpc_client_example.py` | Send text to Fletcher for embedding (HTTP/2 + CBOR) |
| `arrow_flight_client.py` | Vector search on Longbow (Arrow Flight) |

## Documentation

For detailed usage and options, see **[docs/scripts.md](../docs/scripts.md)**.
