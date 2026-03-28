#!/bin/bash
# Full benchmark for embedding engines with larger batches

set -e
cd /home/rsd/REPOS/longbow-fletcher

echo "=== Full Embedding Benchmark ==="
echo ""

# Build fletcher
echo "Building fletcher..."
go build -o fletcher ./cmd/fletcher

# Generate 100 test texts of varying lengths
NUM_TEXTS=100
TEXT_FILE=$(mktemp)
for i in $(seq 1 $NUM_TEXTS); do
    echo "The quick brown fox jumps over the lazy dog number $i with some additional text for variety"
done > "$TEXT_FILE"

echo "Testing with $NUM_TEXTS texts"
echo ""

echo "=== Fletcher (bert-tiny, CPU, batch=$NUM_TEXTS) ==="
start=$(date +%s.%N)
./fletcher encode "$TEXT_FILE" > /dev/null 2>&1
end=$(date +%s.%N)
dur=$(echo "$end - $start" | bc)
echo "Time: ${dur}s"
echo "Throughput: $(echo "scale=1; $NUM_TEXTS / $dur" | bc) texts/sec"
fletcher_throughput=$(echo "scale=1; $NUM_TEXTS / $dur" | bc)
echo ""

echo "=== Ollama (nomic-embed-text, batch=$NUM_TEXTS) ==="
# Ollama doesn't support batch embedding via REST, so we test sequential
start=$(date +%s.%N)
for text in $(cat "$TEXT_FILE"); do
    curl -s -X POST http://localhost:11434/api/embeddings \
        -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"$text\"}" > /dev/null 2>&1 &
done
wait
end=$(date +%s.%N)
dur=$(echo "$end - $start" | bc)
echo "Time: ${dur}s (parallel requests)"
echo "Throughput: $(echo "scale=1; $NUM_TEXTS / $dur" | bc) texts/sec"
ollama_throughput=$(echo "scale=1; $NUM_TEXTS / $dur" | bc)
echo ""

echo "=== llama.cpp (e5-small, if available) ==="
if command -v llama-embedding &> /dev/null; then
    TEXT_E5=$(head -1 "$TEXT_FILE")
    start=$(date +%s.%N)
    llama-embedding --embd-e5-small-en-default -p "$TEXT_E5" 2>/dev/null | head -1
    end=$(date +%s.%N)
    dur=$(echo "$end - $start" | bc)
    echo "Time: ${dur}s"
    echo "Throughput: $(echo "scale=1; 1 / $dur" | bc) texts/sec"
else
    echo "llama-embedding not available"
fi

echo ""
echo "=== Summary ==="
echo "Fletcher (bert-tiny):  $fletcher_throughput texts/sec"
echo "Ollama (nomic-embed):  $ollama_throughput texts/sec"

rm -f "$TEXT_FILE"
echo ""
echo "=== Done ==="
