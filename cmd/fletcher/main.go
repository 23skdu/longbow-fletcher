package main

import (
	"flag"
	"fmt"
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
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
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
	listenAddr  = flag.String("listen", "", "Address to listen on for HTTP Server (e.g. :8080)")
	flightAddr  = flag.String("flight", "", "Address to listen on for Flight Server (e.g. :9090)")
	maxConcurrent = flag.Int("max-concurrent", 16384, "Maximum number of concurrent sequences to process")
	enableOTel    = flag.Bool("otel", false, "Enable OpenTelemetry tracing (stdout)")
	flagMaxVRAM   = flag.String("max-vram", "4GB", "Maximum VRAM to use for admission control (e.g. 4GB, 512MB)")
	flagTransportFmt = flag.String("transport-fmt", "fp32", "Transport format for embeddings: 'fp32' (default) or 'fp16'")
)

func parseBytes(s string) int64 {
	// Simple parser without external deps
	// 4GB, 100MB, 1024
	if s == "" || s == "0" {
		return 0
	}
	var val int64
	var unit string
	fmt.Sscanf(s, "%d%s", &val, &unit)
	
	switch unit {
	case "GB", "G":
		return val * 1024 * 1024 * 1024
	case "MB", "M":
		return val * 1024 * 1024
	case "KB", "K":
		return val * 1024
	default:
		return val
	}
}

func main() {
	// Initialize logging
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}).With().Caller().Logger()

	flag.Parse()

	if *enableOTel {
		shutdown, err := initTracer()
		if err != nil {
			log.Fatal().Err(err).Msg("Failed to initialize tracer")
		}
		defer shutdown(context.Background())
	}
	
	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatal().Err(err).Msg("Failed to create CPU profile file")
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal().Err(err).Msg("Could not start CPU profile")
		}
		defer pprof.StopCPUProfile()
	}

	embedder, err := embeddings.NewEmbedder(*vocabPath, *weightsPath, *useGPU, *modelType, *precision)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create embedder")
	}

	// Server Mode
	if *listenAddr != "" {
		var fcInterface FlightClientInterface
		if *serverAddr != "" {
			var err error
			fc, err := client.NewFlightClient(*serverAddr)
			if err != nil {
				log.Fatal().Err(err).Msg("Failed to create flight client")
			}
			log.Info().Str("addr", *serverAddr).Msg("Connected to Flight Server")
			fcInterface = fc
		}

		maxVRAMBytes := parseBytes(*flagMaxVRAM)
		log.Info().Str("max_vram", *flagMaxVRAM).Int64("bytes", maxVRAMBytes).Msg("VRAM Admission Control")

		go startServer(*listenAddr, embedder, fcInterface, *datasetName, *maxConcurrent, maxVRAMBytes, *flagTransportFmt)
		if *flightAddr == "" {
			// specific usage for wait
			select {}
		}
	}
	
	if *flightAddr != "" {
		StartFlightServer(*flightAddr, embedder)
		return
	}
	
	if *listenAddr != "" {
		// Was waiting above
		select {}
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
		log.Info().Str("duration", duration.String()).Msg("Starting soak test")
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
			_ = embedder.ProxyEmbedBatch(context.Background(), texts)
			_ = time.Since(iterStart) // Keep timer call but ignore result to silence usage error, or just remove.
			
			totalVectors += int64(len(texts))
			iter++
			
			if iter%10 == 0 {
				elapsed := time.Since(startTime)
				tps := float64(totalVectors) / elapsed.Seconds()
				log.Info().
					Str("elapsed", elapsed.Round(time.Second).String()).
					Int("iter", iter).
					Int64("total_vectors", totalVectors).
					Float64("tps", tps).
					Msg("Soak test progress")
			}
		}
		
		totalElapsed := time.Since(startTime)
		log.Info().
			Int64("total_vectors", totalVectors).
			Dur("total_time", totalElapsed).
			Float64("avg_tps", float64(totalVectors)/totalElapsed.Seconds()).
			Msg("Soak test complete")
		return
	}

	start := time.Now()
	vectors := embedder.ProxyEmbedBatch(context.Background(), texts)
	elapsed := time.Since(start)

	dim := 0
	if len(vectors) > 0 {
		dim = len(vectors) / len(texts)
	}
	log.Info().
		Int("count", len(texts)).
		Dur("elapsed", elapsed).
		Int("dim", dim).
		Float64("tps", float64(len(texts))/elapsed.Seconds()).
		Msg("Embedded sequences")

	// Example: write to Arrow IPC
	// Using Arrow Go library
	pool := memory.NewGoAllocator()
	
	// Define Schema
	// Schema: { "text": utf8, "embedding": fixed_size_list<float32>[128] }
	// Assuming 128 dim for BERT Tiny (fallback if dim is 0)
	if dim == 0 {
		dim = 128
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
		floatBuilder.AppendValues(vectors[i*dim:(i+1)*dim], nil)
	}

	textArr := textBuilder.NewArray()
	defer textArr.Release()
	embedArr := embedBuilder.NewArray()
	defer embedArr.Release()

	rec := array.NewRecordBatch(schema, []arrow.Array{textArr, embedArr}, int64(len(texts)))
	defer rec.Release()

	// If server is provided, send via Flight
	if *serverAddr != "" {
		log.Info().Int("count", len(texts)).Str("server", *serverAddr).Str("dataset", *datasetName).Msg("Sending vectors to Longbow")
		flightClient, err := client.NewFlightClient(*serverAddr)
		if err != nil {
			log.Fatal().Err(err).Msg("Failed to connect to Longbow")
		}
		defer func() {
			if err := flightClient.Close(); err != nil {
				log.Warn().Err(err).Msg("Failed to close flight client")
			}
		}()

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		if err := flightClient.DoPut(ctx, *datasetName, rec); err != nil {
			log.Fatal().Err(err).Msg("Flight DoPut failed")
		}
		log.Info().Msg("Successfully sent embeddings to Longbow")
	} else {
		// Example: write to Arrow IPC to stdout
		err = writeArrowStream(os.Stdout, rec)
		if err != nil {
			log.Warn().Err(err).Msg("Failed to write arrow stream")
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

func initTracer() (func(context.Context) error, error) {
	exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		return nil, err
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceNameKey.String("fletcher"),
		)),
	)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))

	return tp.Shutdown, nil
}
