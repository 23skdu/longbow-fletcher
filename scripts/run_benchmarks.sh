#!/bin/bash
# Benchmark script for Fletcher embeddings
# Runs comprehensive benchmarks for all datatypes and dimensions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "Fletcher Benchmark Suite"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build the benchmark binary
echo -e "${GREEN}[1/5] Building benchmark binary...${NC}"
go build -o bin/benchmark ./scripts/benchmark_datatypes.go

# Run datatype benchmarks
echo -e "${GREEN}[2/5] Running datatype benchmarks...${NC}"
echo "------------------------------------------"
./bin/benchmark -test.bench=BenchmarkDatatypeSizes -test.benchtime=1s

# Run dimension benchmarks  
echo -e "${GREEN}[3/5] Running dimension benchmarks...${NC}"
echo "------------------------------------------"
./bin/benchmark -test.bench=BenchmarkTensorCreation -test.benchtime=1s

# Run operations benchmarks
echo -e "${GREEN}[4/5] Running operations benchmarks...${NC}"
echo "------------------------------------------"
./bin/benchmark -test.bench=BenchmarkTensorOperations -test.benchtime=1s

# Run matrix multiply benchmarks
echo -e "${GREEN}[5/5] Running matrix multiply benchmarks...${NC}"
echo "------------------------------------------"
./bin/benchmark -test.bench=BenchmarkMatrixMultiply -test.benchtime=1s

echo ""
echo -e "${GREEN}========================================"
echo "Benchmark Complete!"
echo "========================================${NC}"
