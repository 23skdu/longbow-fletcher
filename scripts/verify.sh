#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

export PATH="/home/linuxbrew/.linuxbrew/bin:$(go env GOPATH 2>/dev/null || echo "$HOME/go")/bin:$PATH"

echo "=========================================================="
echo " Fletcher Pre-Commit Verification & Quality Gates"
echo "=========================================================="

STAGED_GO_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.go$' || true)
if [ -n "$STAGED_GO_FILES" ]; then
    PKGS=$(echo "$STAGED_GO_FILES" | xargs -n1 dirname | sort -u | sed 's|^\([^/]\)|./\1|')
else
    PKGS="./..."
fi

# Gate 1: formatting + vet + lint
echo "--> Gate 1/5: Formatting, vet, and lint..."
if [ -n "$STAGED_GO_FILES" ]; then
    UNFORMATTED=$(gofmt -l $STAGED_GO_FILES || true)
    if [ -n "$UNFORMATTED" ]; then
        echo "Unformatted files:"; echo "$UNFORMATTED"; exit 1
    fi
fi
go vet $PKGS
if command -v golangci-lint >/dev/null 2>&1; then
    golangci-lint run $PKGS
fi
echo "    ✓ Gate 1 passed"

# Gate 2: 100% test/fuzz presence
echo "--> Gate 2/5: 100% unit/fuzz test presence..."
UNTESTED=$(go test -list '.*' ./... 2>&1 | grep '\[no test files\]' || true)
if [ -n "$UNTESTED" ]; then
    echo "Packages without tests:"; echo "$UNTESTED"; exit 1
fi
./scripts/check_fuzz_coverage.sh
echo "    ✓ Gate 2 passed"

# Gate 3: unit + fuzz tests
echo "--> Gate 3/5: Unit and fuzz tests..."
go test -short ./...
go test -short -run='^Fuzz' ./...
echo "    ✓ Gate 3 passed"

# Gate 4: accuracy gate
echo "--> Gate 4/5: Accuracy gate..."
go test -v ./internal/simd/ -run='Accuracy|Correctness'
echo "    ✓ Gate 4 passed"

# Gate 5: oracle gate
echo "--> Gate 5/5: Oracle gate..."
go test -v ./internal/simd/... ./internal/device/... ./internal/embeddings/... -run='Parity|Oracle'
echo "    ✓ Gate 5 passed"

echo "=========================================================="
echo " ✓ All verification gates passed!"
echo "=========================================================="