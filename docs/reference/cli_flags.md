# CLI Flags Reference

This reference documents the command-line flags available in the `fletcher` binary.

## General

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-help` | `bool` | `false` | Show usage information. |
| `-gpu` | `bool` | `false` | Enable Metal GPU acceleration (macOS only). |
| `-cpuprofile` | `string` | `""` | Write CPU profile to the specified file. |
| `-otel` | `bool` | `false` | Enable OpenTelemetry tracing. |

## Model Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | `bert-tiny` | Model architecture (`bert-tiny`, `nomic-embed-text`). |
| `-vocab` | `vocab.txt` | Path to WordPiece vocabulary file. |
| `-weights` | `bert_tiny.bin` | Path to weights binary. |
| `-precision` | `fp32` | Computation precision (`fp32`, `fp16`). |

## Text Input

| Flag | Description |
|------|-------------|
| `-text` | Single text string to embed. |
| `-input` | Path to JSON file containing array of strings (`["text1", "text2"]`). |
| `-task` | Nomic-specific task prefix (e.g., `search_query`, `search_document`). |

## Server & Forwarding

| Flag | Default | Description |
|------|---------|-------------|
| `-listen` | `""` | HTTP server address (e.g., `:8080`). |
| `-flight` | `""` | Flight server address (e.g., `:9090`). |
| `-server` | `""` | Remote Longbow Flight server address to forward embeddings to. |
| `-dataset` | `fletcher_dataset` | Target dataset name on remote server. |
| `-transport-fmt` | `fp32` | Transport precision (`fp32`, `fp16`). |

## Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `-max-concurrent` | `16384` | Max concurrent sequences admission limit. |
| `-max-vram` | `4GB` | VRAM estimation hint (e.g. `8GB`, `512MB`). |
