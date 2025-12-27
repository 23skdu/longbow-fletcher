# Stage 1: Build
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev openblas-dev

# Copy go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=1 GOOS=linux go build -ldflags="-s -w" -o fletcher ./cmd/fletcher

# Stage 2: Runtime image
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache openblas libstdc++

WORKDIR /app

# Copy the binary
COPY --from=builder /app/fletcher /app/fletcher
COPY --from=builder /app/vocab.txt /app/vocab.txt

ENTRYPOINT ["/app/fletcher"]
