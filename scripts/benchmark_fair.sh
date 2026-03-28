#!/bin/bash
# Fair benchmark - Fletcher vs Ollama

set -e
cd /home/rsd/REPOS/longbow-fletcher

echo "=== Fair Embedding Benchmark ==="
echo ""

# Build fletcher
go build -o fletcher ./cmd/fletcher

TEXT="The quick brown fox"
echo "Test: '$TEXT'"
echo ""

# Warmup
./fletcher encode - <<< "$TEXT" > /dev/null 2>&1
curl -s -X POST http://localhost:11434/api/embeddings \
    -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"$TEXT\"}" > /dev/null 2>&1

echo "=== Fletcher (bert-tiny, CPU) ==="
N=20
start=$(date +%s.%N)
for i in $(seq 1 $N); do ./fletcher encode - <<< "$TEXT" > /dev/null 2>&1; done
end=$(date +%s.%N)
dur=$(echo "$end - $start" | bc)
fletcher_tp=$(echo "scale=1; $N / $dur" | bc)
echo "Throughput: ${fletcher_tp}/s"

echo ""
echo "=== Ollama (nomic-embed-text, GPU) ==="
start=$(date +%s.%N)
for i in $(seq 1 $N); do 
    curl -s -X POST http://localhost:11434/api/embeddings \
        -d "{\"model\":\"nomic-embed-text\",\"prompt\":\"$TEXT\"}" > /dev/null 2>&1
done
end=$(date +%s.%N)
dur=$(echo "$end - $start" | bc)
ollama_tp=$(echo "scale=1; $N / $dur" | bc)
echo "Throughput: ${ollama_tp}/s"

echo ""
echo "=== Results ==="
echo "Fletcher: ${fletcher_tp}/s (bert-tiny, 128-dim, CPU)"
echo "Ollama:   ${ollama_tp}/s (nomic-embed-text, 768-dim, GPU)"
echo ""
echo "Fletcher is $(echo "scale=1; $fletcher_tp / $ollama_tp" | bc)x faster"
