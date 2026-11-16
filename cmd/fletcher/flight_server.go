package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"os"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc/credentials"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/rs/zerolog/log"

	"google.golang.org/grpc"
)

type FletcherFlightServer struct {
	flight.BaseFlightServer
	embedder EmbedderInterface
	alloc    memory.Allocator
}

func NewFletcherFlightServer(embedder EmbedderInterface) *FletcherFlightServer {
	return &FletcherFlightServer{
		embedder: embedder,
		alloc:    memory.NewGoAllocator(),
	}
}

func (s *FletcherFlightServer) DoAction(action *flight.Action, stream flight.FlightService_DoActionServer) error {
	if action != nil && string(action.Type) == "GetModels" {
		models := []string{
			"bert-tiny", "all-MiniLM-L6-v2", "nomic-embed-text", "roberta-base",
			"xlm-roberta-base", "bge-m3", "e5-mistral-7b-instruct",
		}
		b, _ := json.Marshal(models)
		return stream.Send(&flight.Result{Body: b})
	}
	if action != nil && string(action.Type) == "GetDimensions" {
		dims := map[string]int{
			"bert-tiny":              128,
			"all-MiniLM-L6-v2":       384,
			"nomic-embed-text":       768,
			"roberta-base":           768,
			"xlm-roberta-base":       768,
			"bge-m3":                 1024,
			"e5-mistral-7b-instruct": 4096,
		}
		b, _ := json.Marshal(dims)
		return stream.Send(&flight.Result{Body: b})
	}
	return s.BaseFlightServer.DoAction(action, stream)
}

func (s *FletcherFlightServer) DoExchange(stream flight.FlightService_DoExchangeServer) error {
	reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(s.alloc))
	if err != nil {
		log.Error().Err(err).Msg("Failed to create Flight reader")
		return err
	}
	defer reader.Release()

	for reader.Next() {
		if desc := reader.LatestFlightDescriptor(); desc != nil {
			log.Info().
				Str("cmd", string(desc.Cmd)).
				Strs("path", desc.Path).
				Msg("Flight DoExchange descriptor")

			if len(desc.Path) > 0 && desc.Path[0] == "ingest" {
				return fmt.Errorf("ingest not supported over Fletcher embedding service; use Longbow DoPut")
			}
		}

		rec := reader.RecordBatch()
		if rec.NumRows() == 0 {
			continue
		}

		// Expect schema: {row_id: uint64, text: utf8} or {row_id, text, modality}
		colID := rec.Column(0)
		colText := rec.Column(1)

		// Validate types (simple check)
		if colID.DataType().ID() != arrow.UINT64 || (colText.DataType().ID() != arrow.STRING) {
			return fmt.Errorf("invalid schema: expected {row_id: uint64, text: string}")
		}

		if rec.NumCols() >= 3 {
			log.Info().Msg("Flight request includes modality column")
		}

		ids := colID.(*array.Uint64)
		textsArr := colText.(*array.String)

		rowCount := int(rec.NumRows())
		texts := make([]string, rowCount)
		rowIDs := make([]uint64, rowCount)

		for i := 0; i < rowCount; i++ {
			texts[i] = textsArr.Value(i)
			rowIDs[i] = ids.Value(i)
		}

		log.Info().Int("count", rowCount).Msg("DoExchange processing batch")

		// Process batch
		ctx := stream.Context()
		ch := s.embedder.EmbedBatch(ctx, texts)

		// Collect all results to reconstruct batch (or stream back as they come?)
		// To keep it simple and efficient, we can buffer results for the batch
		// and send them back. Or better, allow partial sends if Flight supports it.
		// Flight DoExchange is bidirectional streaming.

		// For now, let's collect results to form a proper RecordBatch matching the input size,
		// or at least preserving the ID mapping.
		// EmbedBatch returns chunks.

		// We can send multiple batches back for one input batch.
		// But typically we want to maximize throughput.

		// Collect results
		var writer *flight.Writer

		for chunk := range ch {
			if chunk.Err != nil {
				log.Error().Err(chunk.Err).Msg("Inference error")
				continue
			}

			// Chunk corresponds to texts[chunk.Offset : chunk.Offset+chunk.Count]
			chunkIDs := rowIDs[chunk.Offset : chunk.Offset+chunk.Count]

			// Build Response Batch
			dim := len(chunk.Vectors) / chunk.Count
			outRec := s.buildEmbeddingBatch(chunkIDs, chunk.Vectors, dim)

			if writer == nil {
				writer = flight.NewRecordWriter(stream, ipc.WithSchema(outRec.Schema()))
			}

			if err := writer.Write(outRec); err != nil {
				_ = writer.Close()
				outRec.Release()
				log.Error().Err(err).Msg("Failed to write to stream")
				return err
			}
			outRec.Release()
		}

		if writer != nil {
			_ = writer.Close()
		}
	}

	return reader.Err()
}

// buildEmbeddingBatch creates a record batch from vectors
func (s *FletcherFlightServer) buildEmbeddingBatch(ids []uint64, vectors []float32, dim int) arrow.RecordBatch {
	pool := s.alloc

	// Builders
	idBuilder := array.NewUint64Builder(pool)
	defer idBuilder.Release()
	idBuilder.AppendValues(ids, nil)

	// FixedSizeList Builder
	fslType := arrow.FixedSizeListOf(int32(dim), arrow.PrimitiveTypes.Float32)
	embedBuilder := array.NewFixedSizeListBuilder(pool, int32(dim), arrow.PrimitiveTypes.Float32)
	defer embedBuilder.Release()

	valBuilder := embedBuilder.ValueBuilder().(*array.Float32Builder)

	count := len(ids)
	for i := 0; i < count; i++ {
		embedBuilder.Append(true)
	}
	valBuilder.AppendValues(vectors, nil)

	idArr := idBuilder.NewArray()
	defer idArr.Release()
	embedArr := embedBuilder.NewArray()
	defer embedArr.Release()

	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "row_id", Type: arrow.PrimitiveTypes.Uint64},
			{Name: "embedding", Type: fslType},
		},
		nil,
	)

	return array.NewRecordBatch(schema, []arrow.Array{idArr, embedArr}, int64(count))
}

type TLSConfig struct {
	CertFile string
	KeyFile  string
	CAFile   string
}

func StartFlightServer(addr string, embedder EmbedderInterface, tlsConfig *TLSConfig) {
	// Create the generic Flight Server which manages the GRPC lifecycle
	var opts []grpc.ServerOption
	if tlsConfig != nil && tlsConfig.CertFile != "" && tlsConfig.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(tlsConfig.CertFile, tlsConfig.KeyFile)
		if err != nil {
			log.Fatal().Err(err).Msg("Failed to load TLS credentials")
		}
		tlsCfg := &tls.Config{Certificates: []tls.Certificate{cert}}
		if tlsConfig.CAFile != "" {
			caPEM, err := os.ReadFile(tlsConfig.CAFile)
			if err != nil {
				log.Fatal().Err(err).Msg("Failed to read CA file")
			}
			pool := x509.NewCertPool()
			if !pool.AppendCertsFromPEM(caPEM) {
				log.Fatal().Msg("Failed to parse CA certificate")
			}
			tlsCfg.ClientCAs = pool
			tlsCfg.ClientAuth = tls.RequireAndVerifyClientCert
		}
		creds := credentials.NewTLS(tlsCfg)
		opts = append(opts, grpc.Creds(creds))
		log.Info().Msg("Enabling TLS/mTLS for Flight server")
	}

	opts = append(opts, grpc.StatsHandler(otelgrpc.NewServerHandler()))
	server := flight.NewServerWithMiddleware(nil, opts...)

	// Register our custom service implementation
	server.RegisterFlightService(NewFletcherFlightServer(embedder))

	// Init handles the listener creation internally
	if err := server.Init(addr); err != nil {
		log.Fatal().Err(err).Msg("Failed to init Flight server")
	}

	log.Info().Str("addr", addr).Msg("Starting Fletcher Flight Server")
	if err := server.Serve(); err != nil {
		log.Fatal().Err(err).Msg("Flight server failed")
	}
}
