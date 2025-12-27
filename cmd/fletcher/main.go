package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"time"

	"context"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	
	"github.com/23skdu/longbow-fletcher/internal/client"
	"github.com/23skdu/longbow-fletcher/internal/embeddings"
)

var (
	vocabPath   = flag.String("vocab", "vocab.txt", "Path to vocab file")
	weightsPath = flag.String("weights", "bert_tiny.bin", "Path to weights file")
	cpuProfile  = flag.String("cpuprofile", "", "Write cpu profile to file")
	useGPU      = flag.Bool("gpu", false, "Use Metal GPU acceleration")
	interactive = flag.Bool("interactive", false, "Interactive mode")
	loremIpsum  = flag.Int("lorem", 0, "Generate N lines of lorem ipsum")
	modelType   = flag.String("model", "bert-tiny", "Model type (bert-tiny, nomic-embed-text)")
	precision   = flag.String("precision", "fp32", "Precision (fp32, fp16)")
	duration    = flag.Duration("duration", 0, "Run soak test for specified duration (e.g. 10s, 20m)")
	serverAddr  = flag.String("server", "", "Longbow server address (e.g., localhost:3000)")
	datasetName = flag.String("dataset", "fletcher_dataset", "Target dataset name on server")
	listenAddr  = flag.String("listen", "", "Address to listen on for Server Mode (e.g. :8080)")
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	embedder, err := embeddings.NewEmbedder(*vocabPath, *weightsPath, *useGPU, *modelType, *precision)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
		log.Fatalf("Failed to create embedder: %v", err)
	}

	// Server Mode
	if *listenAddr != "" {
		var fc *client.FlightClient
		if *serverAddr != "" {
			var err error
			fc, err = client.NewFlightClient(*serverAddr)
			if err != nil {
				log.Fatalf("Failed to create flight client: %v", err)
			}
			log.Printf("Connected to Flight Server at %s", *serverAddr)
		}

		startServer(*listenAddr, embedder, fc, *datasetName)
		return
	}

	var texts []string
	if *loremIpsum > 0 {
		texts = generateLorem(*loremIpsum)
	} else if *interactive {
		// interactive mode code...
		texts = []string{"Hello world"}
	} else {
		// Just a simple test
		texts = []string{
			"Hello world",
			"The quick brown fox jumps over the lazy dog",
			"Fletcher is a high performance embedding engine",
		}
	}

	if *duration > 0 {
		fmt.Printf("Starting soak test for %v...\n", *duration)
		if len(texts) == 0 {
			// Default soak payload if not specified
			texts = generateLorem(1000)
		}
		
		startTime := time.Now()
		endTime := startTime.Add(*duration)
		var totalVectors int64
		var iter int
		
		for time.Now().Before(endTime) {
			iterStart := time.Now()
			_ = embedder.EmbedBatch(texts)
			_ = time.Since(iterStart) // Keep timer call but ignore result to silence usage error, or just remove.
			
			totalVectors += int64(len(texts))
			iter++
			
			if iter%10 == 0 {
				elapsed := time.Since(startTime)
				tps := float64(totalVectors) / elapsed.Seconds()
				fmt.Printf("[%v] Completed %d iterations. Total: %d vectors. Current TPS: %.2f\n", 
					elapsed.Round(time.Second), iter, totalVectors, tps)
			}
		}
		
		totalElapsed := time.Since(startTime)
		fmt.Printf("Soak test complete.\n")
		fmt.Printf("Total Vectors: %d\n", totalVectors)
		fmt.Printf("Total Time: %v\n", totalElapsed)
		fmt.Printf("Average Throughput: %.2f vectors/sec\n", float64(totalVectors)/totalElapsed.Seconds())
		return
	}

	start := time.Now()
	vectors := embedder.EmbedBatch(texts)
	elapsed := time.Since(start)

	fmt.Printf("Embedded %d texts in %v\n", len(texts), elapsed)
	if len(texts) > 0 {
		fmt.Printf("Vector dim: %d\n", len(vectors[0]))
		fmt.Printf("Throughput: %.2f vectors/sec\n", float64(len(texts))/elapsed.Seconds())
	}

	// Example: write to Arrow IPC
	// Using Arrow Go library
	pool := memory.NewGoAllocator()
	
	// Define Schema
	// Schema: { "text": utf8, "embedding": fixed_size_list<float32>[128] }
	// Assuming 128 dim for BERT Tiny
	dim := 128
	if len(vectors) > 0 {
		dim = len(vectors[0])
	}
	
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "text", Type: arrow.BinaryTypes.String},
			{Name: "embedding", Type: arrow.FixedSizeListOf(int32(dim), arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	// Build Columns
	textBuilder := array.NewStringBuilder(pool)
	defer textBuilder.Release()
	
	embedBuilder := array.NewFixedSizeListBuilder(pool, int32(dim), arrow.PrimitiveTypes.Float32)
	defer embedBuilder.Release()
	floatBuilder := embedBuilder.ValueBuilder().(*array.Float32Builder)

	// Append Data
	for i, text := range texts {
		textBuilder.Append(text)
		
		embedBuilder.Append(true)
		floatBuilder.AppendValues(vectors[i], nil)
	}

	textArr := textBuilder.NewArray()
	defer textArr.Release()
	embedArr := embedBuilder.NewArray()
	defer embedArr.Release()

	rec := array.NewRecordBatch(schema, []arrow.Array{textArr, embedArr}, int64(len(texts)))
	defer rec.Release()

	// If server is provided, send via Flight
	if *serverAddr != "" {
		fmt.Printf("Sending %d vectors to Longbow at %s [%s]...\n", len(texts), *serverAddr, *datasetName)
		flightClient, err := client.NewFlightClient(*serverAddr)
		if err != nil {
			log.Fatalf("Failed to connect to Longbow: %v", err)
		}
		defer func() {
			if err := flightClient.Close(); err != nil {
				log.Printf("Warning: failed to close flight client: %v", err)
			}
		}()

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		if err := flightClient.DoPut(ctx, *datasetName, rec); err != nil {
			log.Fatalf("Flight DoPut failed: %v", err)
		}
		fmt.Println("Successfully sent embeddings to Longbow.")
	} else {
		// Example: write to Arrow IPC to stdout
		err = writeArrowStream(os.Stdout, rec)
	if err != nil {
			log.Printf("Warning: failed to write arrow stream: %v", err)
		}
	}
}

func writeArrowStream(w *os.File, rec arrow.RecordBatch) error {
	writer := ipc.NewWriter(w, ipc.WithSchema(rec.Schema()))
	if err := writer.Write(rec); err != nil {
		_ = writer.Close()
		return err
	}
	return writer.Close()
}

func generateLorem(n int) []string {
	// Simple lorem ipsum generator
	base := "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	res := make([]string, n)
	for i := 0; i < n; i++ {
		res[i] = fmt.Sprintf("%s %d", base, i)
	}
	return res
}
