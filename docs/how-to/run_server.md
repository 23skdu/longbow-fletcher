# How to Run Fletcher as a Server

Fletcher can operate as a high-performance HTTP and Apache Arrow Flight server, allowing you to integrate embedding generation into your applications.

## Starting the Server

To start the server, use the `-listen` flag. You can also enable GPU acceleration.

```bash
./bin/fletcher -listen :8080 -gpu
```

## API Endpoints

### HTTP Ingestion

**POST** `/ingest`

Accepts a JSON payload containing an array of strings.

**Request:**

```json
{
  "texts": [
    "The quick brown fox",
    "Jumps over the lazy dog"
  ]
}
```

**Response:**
Returns a JSON object confirming processing (embeddings are forwarded to Longbow internally or returned if configured).

### Arrow Flight

If configured with `-flight :9090`, Fletcher accepts Arrow DoPut requests containing a record batch with a `text` column.

## Integration with Longbow

To forward generated embeddings to a persistent Longbow vector database:

```bash
./bin/fletcher -listen :8080 -server localhost:3000 -dataset my_wiki_data
```

- `-server`: Address of the Longbow instance.
- `-dataset`: Target dataset name.

## Configuration Tuning

- **`-max-concurrent`**: Limits concurrent sequences (default 16384). Reduce if running out of memory.
- **`-max-vram`**: Hints admission control about available VRAM (e.g., `8GB`).
