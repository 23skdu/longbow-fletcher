package main

import (
	"fmt"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
)

func main() {
	// Config matching prajjwal1/bert-tiny
	config := model.BertConfig{
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		VocabSize:             30522,
		MaxPositionEmbeddings: 512,
	}

	backend := device.NewMetalBackend()
	bertModel := model.NewBertModelWithBackend(config, backend)

	seqLen := 50 // ~50 tokens per sentence to match Python

	// Benchmark function
	benchmark := func(nVectors int) {
		// Generate input
		batchSize := 64
		numBatches := nVectors / batchSize

		// Warmup
		inputIDs := make([]int, batchSize*seqLen)
		lengths := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			lengths[i] = seqLen
			for j := 0; j < seqLen; j++ {
				inputIDs[i*seqLen+j] = (i*seqLen + j) % 1000
			}
		}
		_ = bertModel.ForwardBatch(inputIDs, lengths)
		backend.Synchronize()

		// Timed run
		start := time.Now()
		for b := 0; b < numBatches; b++ {
			_ = bertModel.ForwardBatch(inputIDs, lengths)
		}
		backend.Synchronize()
		elapsed := time.Since(start)

		throughput := float64(nVectors) / elapsed.Seconds()
		fmt.Printf("Fletcher (Metal) - %d vectors: %.2fs (%.0f vec/s)\n", nVectors, elapsed.Seconds(), throughput)
	}

	fmt.Println("\n--- Fletcher (Metal) Benchmark ---")
	benchmark(10_000)
	benchmark(20_000)
}
