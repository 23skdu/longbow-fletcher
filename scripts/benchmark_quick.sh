#!/bin/bash
# Quick benchmark for embedding engines

set -e
cd /home/rsd/REPOS/longbow-fletcher

echo "=== Embedding Engine Quick Benchmark ==="
echo ""

# Build fletcher
echo "Building fletcher..."
go build -o fletcher ./cmd/fletcher

# Test texts - use more for better throughput measurement
TEXTS=(
"The quick brown fox"
"Artificial intelligence"
"Machine learning"
"Neural networks"
"Deep learning"
"Attention mechanism"
"Transformer models"
"Embedding vectors"
"Natural language processing"
"Computer vision"
)

NUM_TEXTS=10
TEXT_FILE=$(mktemp)
printf '%s\n' "${TEXTS[@]}" > "$TEXT_FILE"

echo "Testing with $NUM_TEXTS texts"
echo ""

echo "=== Fletcher (bert-tiny, CPU) ==="
start=$(date +%s.%N)
./fletcher encode "$TEXT_FILE" > /dev/null 2>&1
end=$(date +%s.%N)
dur=$(echo "$end - $start" | bc)
echo "Time: ${dur}s"
echo "Throughput: $(echo "scale=1; $NUM_TEXTS / $dur" | bc) texts/sec"
echo ""

echo "=== Ollama (nomic-embed-text) ==="
start=$(date +%s.%N)
for text in "${TEXTS[@]}"; do
    curl -s -X POST http://localhost:11434/api/embeddings \
        -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"$text\"}" > /dev/null
done
end=$(date +%s.%N)
dur=$(echo "$end - $start" | bc)
echo "Time: ${dur}s"
echo "Throughput: $(echo "scale=1; $NUM_TEXTS / $dur" | bc) texts/sec"
echo ""

rm -f "$TEXT_FILE"
echo "=== Done ==="
