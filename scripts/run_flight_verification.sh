#!/bin/bash
set -e

# Build fletcher
echo "Building fletcher..."
# export CGO_ENABLED=0
export CGO_LDFLAGS="-framework Accelerate -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph" 
# Adding LDFLAGS explicit just in case, but -tags metal is critical.
go build -tags metal -o bin/fletcher ./cmd/fletcher

# Build verification script
echo "Building verifier..."
# Need to use go run because it's a script in main package but inside scripts dir, 
# relying on internal packages. 
# "go run scripts/verify_flight.go" might fail if it can't find internal modules from there?
# Actually, since it imports github.com/23skdu/longbow-fletcher/internal/client, it should work if run from root.

# Start Server
echo "Starting Fletcher Server..."
./bin/fletcher -model bert-tiny -flight :9090 -gpu=true &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"

cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID || true
}
trap cleanup EXIT

# Wait for server
echo "Waiting for server..."
sleep 5

# Run Verification
echo "Running verification..."
go run scripts/verify_flight.go :9090

echo "Done."
