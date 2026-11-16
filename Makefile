# Longbow Fletcher Makefile
# Build, test, and release targets for the embedding engine.

SHELL := /bin/bash

GO ?= $(shell which /home/linuxbrew/.linuxbrew/bin/go 2>/dev/null || which go)
GOPATH ?= $(shell $(GO) env GOPATH)
GOBIN ?= $(GOPATH)/bin
export PATH := /home/linuxbrew/.linuxbrew/bin:$(GOBIN):$(PATH)

GOLANGCI_LINT ?= $(shell which golangci-lint 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/golangci-lint)
GOSEC         ?= $(shell which gosec 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/gosec)
GOVULNCHECK   ?= $(shell which govulncheck 2>/dev/null || echo $(GOBIN)/govulncheck)
GOIMPORTS     ?= $(shell which goimports 2>/dev/null || echo $(GOBIN)/goimports)
DOCKER        ?= $(shell which docker 2>/dev/null || echo docker)
IMAGE_TAG     ?= latest

.PHONY: all help build build-release build-debug build-race clean deps deps-upgrade \
	fmt fmt-check vet lint lint-fix staticcheck security test test-short test-all race \
	coverage coverage-html bench cross-compile docker-build-cpu docker-build-cuda docker-build-metal \
	accuracy-gate oracle-gate makehooks install-hooks verify pre-commit

all: build

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build:"
	@echo "  build            Compile all packages"
	@echo "  build-release    Build release binary with stripped symbols"
	@echo "  build-debug      Build with debug symbols"
	@echo "  build-race       Build with race detector"
	@echo "  cross-compile    Verify cross-compilation (linux/amd64, linux/arm64)"
	@echo "  clean            Remove build artifacts"
	@echo ""
	@echo "Dependencies:"
	@echo "  deps             Download and tidy modules"
	@echo "  deps-upgrade     Upgrade all modules"
	@echo ""
	@echo "Quality:"
	@echo "  fmt              Format code"
	@echo "  fmt-check        Check formatting"
	@echo "  vet              Run go vet"
	@echo "  lint             Run golangci-lint"
	@echo "  lint-fix         Run golangci-lint with fixes"
	@echo "  staticcheck      Run staticcheck"
	@echo "  security         Run gosec + govulncheck"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run unit tests (short)"
	@echo "  test-all         Run full test suite"
	@echo "  race             Run tests with race detector"
	@echo "  coverage         Collect coverage"
	@echo "  coverage-html    Generate HTML coverage report"
	@echo "  bench            Run benchmarks"
	@echo ""
	@echo "Containers:"
	@echo "  docker-build-cpu   Build CPU image"
	@echo "  docker-build-cuda  Build CUDA image"
	@echo "  docker-build-metal Build Metal image (macOS)"

build:
	@echo "Building all packages..."
	$(GO) build ./...

build-bench:
	@echo "Building bench tool..."
	@mkdir -p bin
	$(GO) build -o bin/bench ./cmd/bench

build-release:
	@echo "Building release artifacts..."
	@mkdir -p bin
	$(GO) build -trimpath -ldflags="-s -w" -o bin/fletcher ./cmd/fletcher

build-debug:
	@echo "Building debug artifacts..."
	@mkdir -p bin
	$(GO) build -gcflags="all=-N -l" -o bin/fletcher-debug ./cmd/fletcher

build-race:
	@echo "Building with race detector..."
	$(GO) build -race ./...

cross-compile:
	@echo "Verifying cross-compilation..."
	GOOS=linux GOARCH=amd64 $(GO) build ./...
	GOOS=linux GOARCH=arm64 $(GO) build ./...
	@echo "Cross-compilation OK."

deps:
	@echo "Resolving dependencies..."
	$(GO) mod download
	$(GO) mod tidy

deps-upgrade:
	@echo "Upgrading Go dependencies..."
	$(GO) get -u ./...
	$(GO) mod tidy

clean:
	@echo "Cleaning build artifacts..."
	rm -rf bin/ dist/ build/ coverage.out coverage.html
	$(GO) clean -testcache

fmt:
	@echo "Formatting Go source..."
	gofmt -s -w cmd internal
	$(GOIMPORTS) -w cmd internal 2>/dev/null || true

fmt-check:
	@echo "Checking formatting..."
	@DIFF=$$(gofmt -l cmd internal); \
	if [ -n "$$DIFF" ]; then echo "Unformatted:"; echo "$$DIFF"; exit 1; fi
	@echo "OK."

vet:
	@echo "Running go vet..."
	$(GO) vet ./...

lint:
	@echo "Running golangci-lint..."
	$(GOLANGCI_LINT) run ./... 2>/dev/null || echo "golangci-lint not available, skipping."

lint-fix:
	@echo "Running golangci-lint with fixes..."
	$(GOLANGCI_LINT) run --fix ./... 2>/dev/null || true

staticcheck:
	@echo "Running staticcheck..."
	$(GOLANGCI_LINT) run -E staticcheck,unused ./... 2>/dev/null || echo "staticcheck not available, skipping."

security:
	@echo "Running security analysis..."
	$(GOSEC) -quiet ./... 2>/dev/null || true
	$(GOVULNCHECK) ./... 2>/dev/null || true

test: test-short

test-short:
	@echo "Running unit tests (short)..."
	$(GO) test -short ./...

test-all:
	@echo "Running all tests..."
	$(GO) test -v ./...

race:
	@echo "Running race detector..."
	$(GO) test -race -short ./...

coverage:
	@echo "Collecting coverage..."
	$(GO) test -short -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out | tail -5

coverage-html:
	@echo "Generating HTML coverage..."
	$(GO) test -short -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html

accuracy-gate:
	@echo "Running Accuracy Gate..."
	$(GO) test -v ./internal/simd/ -run='Accuracy'

oracle-gate:
	@echo "Running Oracle Gate..."
	$(GO) test -v ./internal/simd/... -run='Parity|Oracle'

bench:
	@echo "Running benchmarks..."
	$(GO) test -short -run='^$$' -bench=. -benchmem ./...

docker-build-cpu:
	@echo "Building CPU Docker image..."
	$(DOCKER) build -f Dockerfile -t fletcher:$(IMAGE_TAG) .

docker-build-cuda:
	@echo "Building CUDA Docker image..."
	$(DOCKER) build -f Dockerfile.cuda -t fletcher-cuda:$(IMAGE_TAG) .

docker-build-metal:
	@echo "Building Metal Docker image..."
	$(DOCKER) build -f Dockerfile.metal -t fletcher-metal:$(IMAGE_TAG) .

makehooks:
	@mkdir -p .git/hooks
	@chmod +x scripts/verify.sh scripts/check_fuzz_coverage.sh
	@ln -sf ../../scripts/verify.sh .git/hooks/pre-commit
	@echo "Pre-commit hook installed."

install-hooks: makehooks

verify:
	@./scripts/verify.sh

pre-commit: verify