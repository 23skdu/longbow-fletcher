package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/embeddings"
)

type BenchResult struct {
	Model            string  `json:"model"`
	Precision        string  `json:"precision"`
	GPU              bool    `json:"gpu"`
	BatchSize        int     `json:"batch_size"`
	Iterations       int     `json:"iterations"`
	TotalVectors     int     `json:"total_vectors"`
	TotalSeconds     float64 `json:"total_seconds"`
	VectorsPerSecond float64 `json:"vectors_per_second"`
	MeanLatencyMs    float64 `json:"mean_latency_ms"`
	P50LatencyMs     float64 `json:"p50_latency_ms"`
	P95LatencyMs     float64 `json:"p95_latency_ms"`
	P99LatencyMs     float64 `json:"p99_latency_ms"`
	HeapAllocBytes   uint64  `json:"heap_alloc_bytes"`
	HeapObjects      uint64  `json:"heap_objects"`
	Goroutines       int     `json:"goroutines"`
}

func main() {
	vocabPath := flag.String("vocab", "vocab.txt", "vocab file")
	weightsPath := flag.String("weights", "bert_tiny.safetensors", "weights file")
	modelType := flag.String("model", "bert-tiny", "model type")
	precision := flag.String("precision", "fp32", "precision")
	gpu := flag.Bool("gpu", false, "use GPU")
	batchSize := flag.Int("batch", 32, "batch size")
	iters := flag.Int("iters", 50, "iterations")
	warmup := flag.Int("warmup", 5, "warmup iterations")
	textLen := flag.Int("text-len", 16, "words per text")
	output := flag.String("output", "", "output JSON file")
	flag.Parse()

	embedder, err := embeddings.NewEmbedder(*vocabPath, *weightsPath, *gpu, *modelType, *precision)
	if err != nil {
		fmt.Fprintln(os.Stderr, "new embedder:", err)
		os.Exit(1)
	}

	texts := make([]string, *batchSize)
	words := []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "fletcher", "embedding", "vector", "search"}
	for i := range texts {
		n := *textLen
		b := make([]string, n)
		for j := range b {
			b[j] = words[rand.Intn(len(words))]
		}
		texts[i] = join(b, " ")
	}

	// warmup
	for i := 0; i < *warmup; i++ {
		_ = embedder.ProxyEmbedBatch(context.Background(), texts)
	}

	latencies := make([]float64, 0, *iters)
	start := time.Now()
	for i := 0; i < *iters; i++ {
		t0 := time.Now()
		_ = embedder.ProxyEmbedBatch(context.Background(), texts)
		latencies = append(latencies, float64(time.Since(t0).Microseconds())/1000.0)
	}
	total := time.Since(start)

	sort.Float64s(latencies)
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	res := BenchResult{
		Model:            *modelType,
		Precision:        *precision,
		GPU:              *gpu,
		BatchSize:        *batchSize,
		Iterations:       *iters,
		TotalVectors:     *batchSize * *iters,
		TotalSeconds:     total.Seconds(),
		VectorsPerSecond: float64(*batchSize**iters) / total.Seconds(),
		MeanLatencyMs:    mean(latencies),
		P50LatencyMs:     percentile(latencies, 0.50),
		P95LatencyMs:     percentile(latencies, 0.95),
		P99LatencyMs:     percentile(latencies, 0.99),
		HeapAllocBytes:   mem.HeapAlloc,
		HeapObjects:      mem.HeapObjects,
		Goroutines:       runtime.NumGoroutine(),
	}

	b, _ := json.MarshalIndent(res, "", "  ")
	if *output != "" {
		_ = os.WriteFile(*output, b, 0644)
	}
	fmt.Println(string(b))
}

func join(ss []string, sep string) string {
	out := ""
	for i, s := range ss {
		if i > 0 {
			out += sep
		}
		out += s
	}
	return out
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	idx := int(float64(len(xs)-1) * p)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(xs) {
		idx = len(xs) - 1
	}
	return xs[idx]
}
