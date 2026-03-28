#!/bin/bash
# Benchmark script: Fletcher vs Ollama vs llama.cpp embedding engines

set -e

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BENCHMARK_DIR/.."

echo "=== Embedding Engine Benchmark ==="
echo ""

# Test data - variety of text lengths
TEXTS=(
    "The quick brown fox jumps over the lazy dog"
    "Artificial intelligence is transforming technology and society"
    "Machine learning models require careful optimization and tuning to achieve best performance"
    "The attention mechanism has become a fundamental component of modern neural network architectures"
)

# Create temp files
TEMP_TEXT=$(mktemp)
TEMP_OUT=$(mktemp)
trap "rm -f $TEMP_TEXT $TEMP_OUT" EXIT

# Join texts with newlines
printf '%s\n' "${TEXTS[@]}" > "$TEMP_TEXT"
NUM_TEXTS=${#TEXTS[@]}
echo "Test: $NUM_TEXT strings"

echo ""
echo "=== 1. Fletcher (bert-tiny, CPU) ==="
echo "---"

# Build fletcher if needed
if [ ! -f "./fletcher" ]; then
    echo "Building fletcher..."
    cd /home/rsd/REPOS/longbow-fletcher
    go build -o fletcher ./cmd/fletcher 2>/dev/null || true
    cd scripts
fi

if [ -f "../fletcher" ]; then
    cd ..
    time ./fletcher encode "$TEMP_TEXT" 2>&1 | tail -3
    echo ""
else
    echo "SKIP: fletcher binary not found"
fi

echo ""
echo "=== 2. Ollama (nomic-embed-text) ==="
echo "---"

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    for text in "${TEXTS[@]}"; do
        curl -s -X POST http://localhost:11434/api/embeddings \
            -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"$text\"}" > /dev/null
    done
    echo "Embedded $NUM_TEXTS texts via Ollama (nomic-embed-text)"
    echo ""
else
    echo "SKIP: Ollama not running"
fi

echo ""
echo "=== 3. llama.cpp (e5-small-v2 default) ==="
echo "---"

# Try default e5-small model (downloads if not cached)
EMBD_OUT=$(mktemp)
trap "rm -f $EMBD_OUT" EXIT

# Run embedding with default e5-small-en model
if command -v llama-embedding &> /dev/null; then
    # Download and run with default e5-small-en model
    echo "Running llama-embedding with e5-small-en (first run may download)..."
    timeout 120 llama-embedding \
        --embd-e5-small-en-default \
        -p "test" 2>&1 | head -5 || echo "llama-embedding failed or timed out"
    
    # Try with a local model if available
    if [ -f "models/nomic-embed-text-q4_k_m.gguf" ]; then
        echo "Using local nomic-embed-text model..."
        llama-embedding -m models/nomic-embed-text-q4_k_m.gguf --pooling mean \
            -p "$(head -1 $TEMP_TEXT)" 2>&1 | head -5
    fi
else
    echo "SKIP: llama-embedding not installed"
fi

echo ""
echo "=== Benchmark Complete ==="
