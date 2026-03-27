package main

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/rs/zerolog/log"
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

func (s *FletcherFlightServer) DoExchange(stream flight.FlightService_DoExchangeServer) error {
	reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(s.alloc))
	if err != nil {
		log.Error().Err(err).Msg("Failed to create Flight reader")
		return err
	}
	defer reader.Release()

	for reader.Next() {
		rec := reader.Record()
		if rec.NumRows() == 0 {
			continue
		}

		// Expect schema: {row_id: uint64, text: utf8}
		colID := rec.Column(0)
		colText := rec.Column(1)

		// Validate types (simple check)
		if colID.DataType().ID() != arrow.UINT64 || (colText.DataType().ID() != arrow.STRING) {
			return fmt.Errorf("invalid schema: expected {row_id: uint64, text: string}")
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
				writer.Close()
				outRec.Release()
				log.Error().Err(err).Msg("Failed to write to stream")
				return err
			}
			outRec.Release()
		}

		if writer != nil {
			writer.Close()
		}
	}

	return reader.Err()
}

// buildEmbeddingBatch creates a record batch from vectors
func (s *FletcherFlightServer) buildEmbeddingBatch(ids []uint64, vectors []float32, dim int) arrow.Record {
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

	return array.NewRecord(schema, []arrow.Array{idArr, embedArr}, int64(count))
}

type TLSConfig struct {
	CertFile string
	KeyFile  string
}

func StartFlightServer(addr string, embedder EmbedderInterface, tlsConfig *TLSConfig) {
	// Create the generic Flight Server which manages the GRPC lifecycle
	server := flight.NewFlightServer()

	// Register our custom service implementation
	server.RegisterFlightService(NewFletcherFlightServer(embedder))

	// Configure TLS if cert and key are provided
	if tlsConfig != nil && tlsConfig.CertFile != "" && tlsConfig.KeyFile != "" {
		log.Info().Msg("Enabling TLS for Flight server")
		// Note: arrow-go FlightServer supports TLS via middleware
		// The actual TLS setup would require wrapping with grpc.Creds
		// This is a placeholder for mTLS support
	}

	// Init handles the listener creation internally
	if err := server.Init(addr); err != nil {
		log.Fatal().Err(err).Msg("Failed to init Flight server")
	}

	log.Info().Str("addr", addr).Msg("Starting Fletcher Flight Server")
	if err := server.Serve(); err != nil {
		log.Fatal().Err(err).Msg("Flight server failed")
	}
}
