package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}
	// Config matching prajjwal1/bert-tiny
	config := model.BertConfig{
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		VocabSize:             30522,
		MaxPositionEmbeddings: 512,
	}

	backend := device.NewMetalBackendFP16() // FP16 with bulk weight upload
	bertModel := model.NewBertModelWithBackend(config, backend)

	seqLen := 50 // ~50 tokens per sentence to match Python

	// Benchmark function
	benchmark := func(nVectors int) {
		// Generate input
		batchSize := 128 // Reduced batch size for safety / lower latency
		numBatches := nVectors / batchSize
		if nVectors%batchSize != 0 {
			numBatches++
		}

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
			
			// Frequent synchronization and GC to prevent VRAM exhaustion/Lockup
			if b % 10 == 0 {
				backend.Synchronize() 
				runtime.GC() // Force release of Metal buffers held by finalizers
			}
		}
		backend.Synchronize()
		elapsed := time.Since(start)

		throughput := float64(nVectors) / elapsed.Seconds()
		fmt.Printf("Fletcher (Metal) - %d vectors: %.2fs (%.0f vec/s)\n", nVectors, elapsed.Seconds(), throughput)
	}

	fmt.Println("\n--- Fletcher (Metal) Benchmark ---")
	benchmark(10_000)
	benchmark(100_000)
	benchmark(500_000)
	// benchmark(1_000_000) // Skipping 1M in single run to avoid test harness GC issues
}
