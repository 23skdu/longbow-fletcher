# Dockerfile.metal - Build Fletcher with Metal GPU support
# NOTE: Metal (Apple GPU) requires macOS to build. This Dockerfile is provided
# for reference and only works on macOS with Xcode and Metal SDK installed.
# For cross-platform builds, use Dockerfile (CPU) or Dockerfile.cuda (NVIDIA).

# Stage 1: Build (macOS only - requires Xcode and Metal SDK)
FROM --platform=darwin/arm64 golang:1.26 AS builder

WORKDIR /app

# Copy go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary with Metal support (macOS only)
RUN CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -tags metal -ldflags="-s -w" -o fletcher ./cmd/fletcher

# Stage 2: Runtime image (macOS only)
FROM --platform=darwin/arm64 macos:latest

WORKDIR /app

# Copy the binary and vocab
COPY --from=builder /app/fletcher /app/fletcher
COPY --from=builder /app/vocab.txt /app/vocab.txt

ENTRYPOINT ["/app/fletcher"]
