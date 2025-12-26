# Stage 1: Build
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install build dependencies
# gcc and musl-dev are needed for CGO
# openblas-dev is needed for linking against OpenBLAS
RUN apk add --no-cache git gcc musl-dev openblas-dev

# Copy go module files first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary with CGO enabled
# We don't use -static here because we want to link against shared OpenBLAS
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o fletcher ./cmd/fletcher

# Stage 2: Runtime image
# We need a small distro with a package manager to install libopenblas
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache openblas libstdc++

WORKDIR /app

# Copy the binary
COPY --from=builder /app/fletcher /app/fletcher

# Copy default vocab if available
COPY --from=builder /app/vocab.txt /app/vocab.txt

ENTRYPOINT ["/app/fletcher"]
