# Fletcher Prometheus Metrics

Fletcher exposes real-time performance and resource utilization metrics via a Prometheus-compatible endpoint.

## Endpoint

By default, metrics are available at:
`GET http://<fletcher-host>:8080/metrics`

---

## Exported Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `fletcher_vectors_processed_total` | Counter | Total number of vectors embedded since process start. |
| `fletcher_request_duration_seconds` | Histogram | Time spent processing `/encode` requests. Includes decomposition of tokenization and inference. |
| `fletcher_vram_allocated_bytes` | Gauge | Current amount of VRAM allocated by the backend (Metal/CUDA) or Heap (CPU). |

### Detailed Metric Descriptions

#### `fletcher_vectors_processed_total`

- **Description**: This counter tracks the total throughput of the engine.
- **Usage**: Calculate throughput rate using `rate(fletcher_vectors_processed_total[1m])`.

#### `fletcher_request_duration_seconds`

- **Description**: A histogram tracking the latency of embedding requests.
- **Buckets**: Uses default Prometheus buckets (0.005s to 10s).
- **Usage**: Calculate P99 latency: `histogram_quantile(0.99, sum(rate(fletcher_request_duration_seconds_bucket[5m])) by (le))`.

#### `fletcher_vram_allocated_bytes`

- **Description**: Reports the current memory footprint of the model and its intermediate buffers on the hardware accelerator.
- **Backends**:
  - **Metal**: Reports specific hardware allocation sizes.
  - **CPU**: Reports current Go heap allocation if using the CPU backend.
- **Usage**: Monitor for memory leaks or capacity planning.

---

## Grafana Integration

These metrics are designed to be used with the pre-defined Grafana dashboard located at `grafana/fletcher.json`.
