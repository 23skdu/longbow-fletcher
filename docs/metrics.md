# Prometheus Metrics

Fletcher exposes the following Prometheus metrics at `/metrics` (if configured with a metrics server).

## GPU / Device Metrics

| Name | Type | Description | Labels |
| :--- | :--- | :--- | :--- |
| `fletcher_gpu_throughput` | Gauge | GPU throughput in sequences per second | `device` |
| `fletcher_gpu_batch_time_seconds` | Gauge | Last batch processing time in seconds | `device` |
| `fletcher_gpu_batch_count_total` | Counter | Total number of batches processed | `device` |
| `fletcher_gpu_sequences_total` | Counter | Total number of sequences processed | `device` |
| `fletcher_gpu_tokens_total` | Counter | Total number of tokens processed | `device` |
| `fletcher_gpu_weight` | Gauge | Current load balancing weight (based on throughput) | `device` |
| `fletcher_gpu_layer_duration_seconds` | Histogram | Time spent in specific model layers | `layer_type`, `device` |

## Memory & Caching

| Name | Type | Description | Labels |
| :--- | :--- | :--- | :--- |
| `fletcher_metal_pool_hits_total` | Counter | Total successful buffer pool retrievals | - |
| `fletcher_metal_pool_misses_total` | Counter | Total buffer pool misses (allocations) | - |
| `fletcher_metal_pool_size_bytes` | Gauge | Current total size of buffers in pool | - |
| `fletcher_metal_pool_buffers_count` | Gauge | Current number of buffers in pool | - |
| `fletcher_cache_hits_total` | Counter | Embedding cache hits | - |
| `fletcher_cache_misses_total` | Counter | Embedding cache misses | - |

## Tokenization & Request Processing

| Name | Type | Description | Labels |
| :--- | :--- | :--- | :--- |
| `fletcher_tokenization_duration_seconds` | Histogram | Time spent in tokenization | - |
| `fletcher_tokenization_throughput` | Gauge | Tokenization throughput (tokens/sec) | - |
| `fletcher_batch_size_distribution` | Histogram | Distribution of processed batch sizes | - |
| `fletcher_output_invalid_total` | Counter | Total invalid outputs (NaNs) | - |
| `fletcher_panics_total` | Counter | Total recovered panics in worker routines | - |
