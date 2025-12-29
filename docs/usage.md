# Fletcher Usage Guide

`fletcher` is a high-performance embedding engine and client. This guide details the command-line flags available for configuring the server, client, and integration with remote Longbow instances.

## Basic Usage

```bash
# General
./fletcher [flags]

# Start Server
./fletcher -listen :8080 -gpu

# Run Client (Soak Test)
./fletcher -server localhost:3000 -duration 30s
```

## Flags Reference

### General Configuration

| Flag          | Type     | Default       | Description                                      |
|---------------|----------|---------------|--------------------------------------------------|
| `-cpuprofile` | `string` | `""`          | Write CPU profile to the specified file.         |
| `-otel`       | `bool`   | `false`       | Enable OpenTelemetry tracing (outputs to stdout).|
| `-gpu`        | `bool`   | `false`       | Enable Metal GPU acceleration (macOS only).      |

### Model Configuration

Control which model is loaded and how it is executed.

| Flag          | Type     | Default       | Description                                                 |
|---------------|----------|---------------|-------------------------------------------------------------|
| `-model`      | `string` | `"bert-tiny"` | Model type to load. Options: `bert-tiny`, `nomic-embed-text`. |
| `-vocab`      | `string` | `"vocab.txt"` | Path to the vocabulary file.                                |
| `-weights`    | `string` | `"bert_tiny.bin"` | Path to the weights file.                                   |
| `-precision`  | `string` | `"fp32"`      | **Computation** precision for Metal backend. Options: `fp32`, `fp16`. |

### Server Mode

Flags for running Fletcher as an HTTP or Arrow Flight server.

| Flag             | Type     | Default       | Description                                                                 |
|------------------|----------|---------------|-----------------------------------------------------------------------------|
| `-listen`        | `string` | `""`          | Address to listen on for the HTTP API (e.g., `:8080`).                      |
| `-flight`        | `string` | `""`          | Address to listen on for the Arrow Flight server (e.g., `:9090`).           |
| `-max-concurrent`| `int`    | `16384`       | Maximum number of concurrent sequences to process (Concurrency Admission Control). |
| `-max-vram`      | `string` | `"4GB"`       | Maximum VRAM usage estimation for admission control (e.g., `4GB`, `512MB`). |

### Client / Soak Test Mode

Flags for generating synthetic load or running interactive sessions.

| Flag           | Type       | Default | Description                                                                 |
|----------------|------------|---------|-----------------------------------------------------------------------------|
| `-interactive` | `bool`     | `false` | Start interactive mode (REPL).                                              |
| `-lorem`       | `int`      | `0`     | Generate N lines of Lorem Ipsum text for testing.                           |
| `-duration`    | `duration` | `0`     | Run a soak test for the specified duration (e.g., `10s`, `20m`).            |

### Remote Longbow Integration (Forwarding)

Flags for configuring where Fletcher sends the generated embeddings.

| Flag             | Type     | Default            | Description                                                                                     |
|------------------|----------|--------------------|-------------------------------------------------------------------------------------------------|
| `-server`        | `string` | `""`               | Address of the remote Longbow/Arrow Flight server (e.g., `localhost:3000`). If set, embeddings are forwarded here. |
| `-dataset`       | `string` | `"fletcher_dataset"`| Target dataset name on the remote server.                                                      |
| `-transport-fmt` | `string` | `"fp32"`           | **Transport** format for embeddings. Options: `fp32` (default), `fp16`. `fp16` reduces bandwidth by 50%. |

## Examples

**1. Running a high-performance server with GPU and FP16 transport:**

```bash
./fletcher -listen :8080 -gpu -precision fp16 -transport-fmt fp16 -max-vram 8GB
```

**2. Running a soak test against a remote Longbow instance:**

```bash
./fletcher -server localhost:3000 -dataset my_test -duration 1m -lorem 1000
```

**3. Profiling the application locally:**

```bash
./fletcher -cpuprofile cpu.prof -lorem 5000
go tool pprof cpu.prof
```
