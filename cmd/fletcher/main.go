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
	serverAddr  = flag.String("server", "", "Longbow server address (e.g., localhost:3000)")
	datasetName = flag.String("dataset", "fletcher_dataset", "Target dataset name on server")
)

func main() {
	flag.Parse()

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	embedder, err := embeddings.NewEmbedder(*vocabPath, *weightsPath, *useGPU, *modelType)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
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

	builder := array.NewRecordBuilder(pool, schema)
	defer builder.Release()

	textBuilder := builder.Field(0).(*array.StringBuilder)
	embedBuilder := builder.Field(1).(*array.FixedSizeListBuilder)
	floatBuilder := embedBuilder.ValueBuilder().(*array.Float32Builder)

	// Append Data
	for i, text := range texts {
		textBuilder.Append(text)
		
		embedBuilder.Append(true)
		// vectors[i] is already []float32 now!
		floatBuilder.AppendValues(vectors[i], nil)
	}

	rec := builder.NewRecord()
	defer rec.Release()

	// If server is provided, send via Flight
	if *serverAddr != "" {
		fmt.Printf("Sending %d vectors to Longbow at %s [%s]...\n", len(texts), *serverAddr, *datasetName)
		flightClient, err := client.NewFlightClient(*serverAddr)
		if err != nil {
			log.Fatalf("Failed to connect to Longbow: %v", err)
		}
		defer flightClient.Close()

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
			// If stdout is closed or something
		}
	}
}

func writeArrowStream(w *os.File, rec arrow.Record) error {
	writer := ipc.NewWriter(w, ipc.WithSchema(rec.Schema()))
	defer writer.Close()
	
	return writer.Write(rec)
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
