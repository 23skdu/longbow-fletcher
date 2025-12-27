# Stage 1: Build
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install build dependencies
# We need gcc and frameworks for Metal bridging (though Metal usually requires macOS to build)
RUN apk add --no-cache git gcc musl-dev

# Copy go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary with metal tag
RUN CGO_ENABLED=1 GOOS=darwin go build -tags metal -ldflags="-s -w" -o fletcher ./cmd/fletcher

# Stage 2: Runtime image
# Note: This image will likely only run on macOS containers or via virtualization 
# that supports Metal.
FROM alpine:latest

WORKDIR /app

# Copy the binary and vocab
COPY --from=builder /app/fletcher /app/fletcher
COPY --from=builder /app/vocab.txt /app/vocab.txt

ENTRYPOINT ["/app/fletcher"]
