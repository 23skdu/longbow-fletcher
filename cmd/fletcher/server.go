package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/fxamacker/cbor/v2"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rs/zerolog/log"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"golang.org/x/sync/semaphore"
	"sync"

	"github.com/23skdu/longbow-fletcher/internal/embeddings"
)

var (
	vectorsProcessed = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fletcher_vectors_processed_total",
		Help: "The total number of vectors embedded",
	})
	
	requestDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "fletcher_request_duration_seconds",
		Help:    "Time spent processing encode requests",
		Buckets: prometheus.DefBuckets,
	})

	// VRAM metrics will be registered dynamically to capture embedder closure
)

type EmbedderInterface interface {
	EmbedBatch(ctx context.Context, texts []string) <-chan embeddings.StreamResult
	ProxyEmbedBatch(ctx context.Context, texts []string) []float32
	GetVRAMUsage() (allocated int64, total int64)
}

type FlightClientInterface interface {
	DoPut(ctx context.Context, datasetName string, record arrow.RecordBatch) error
	Close() error
}

type Server struct {
	embedder     EmbedderInterface
	flightClient FlightClientInterface
	datasetName  string
	alloc        memory.Allocator
	sbPool       sync.Pool
	sem          *semaphore.Weighted
}

func NewServer(embedder EmbedderInterface, fc FlightClientInterface, dataset string, maxConcurrent int) *Server {
	return &Server{
		embedder:     embedder,
		flightClient: fc,
		datasetName:  dataset,
		alloc:        memory.NewGoAllocator(),
		sbPool: sync.Pool{
			New: func() interface{} {
				return array.NewStringBuilder(memory.DefaultAllocator)
			},
		},
		sem: semaphore.NewWeighted(int64(maxConcurrent)),
	}
}

func startServer(addr string, embedder EmbedderInterface, fc FlightClientInterface, dataset string, maxConcurrent int) {
	srv := NewServer(embedder, fc, dataset, maxConcurrent)

	// Register VRAM metrics
	prometheus.MustRegister(prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Name: "fletcher_vram_allocated_bytes",
			Help: "Current VRAM allocated by the backend",
		},
		func() float64 {
			alloc, _ := embedder.GetVRAMUsage()
			return float64(alloc)
		},
	))

	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/encode", srv.handleEncode)
	http.HandleFunc("/encode/arrow", srv.handleEncodeArrow)
	
	http.HandleFunc("/health", srv.handleHealth)

	log.Info().Str("addr", addr).Msg("Starting Fletcher Server")
	if fc != nil {
		log.Info().Msg("Forwarding to Longbow at specified server address")
	}

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal().Err(err).Msg("Server failed")
	}
}

var tracer = otel.Tracer("fletcher-server")

func (s *Server) handleEncode(w http.ResponseWriter, r *http.Request) {
	ctx, span := tracer.Start(r.Context(), "handleEncode")
	defer span.End()

	start := time.Now()
	defer func() {
		requestDuration.Observe(time.Since(start).Seconds())
	}()

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var texts []string
	decoder := cbor.NewDecoder(r.Body)
	if err := decoder.Decode(&texts); err != nil {
		span.RecordError(err)
		http.Error(w, fmt.Sprintf("Bad Request (CBOR decode): %v", err), http.StatusBadRequest)
		return
	}

	if len(texts) == 0 {
		w.WriteHeader(http.StatusOK)
		return
	}

	span.SetAttributes(
		attribute.Int("sequence_count", len(texts)),
	)

	// Admission Control
	weight := int64(len(texts))
	if err := s.sem.Acquire(ctx, weight); err != nil {
		log.Error().Err(err).Msg("Failed to acquire semaphore")
		http.Error(w, "Server busy", http.StatusServiceUnavailable)
		return
	}
	defer s.sem.Release(weight)

	// 2 & 3. Embed and Forward (Pipelined)
	ch := s.embedder.EmbedBatch(ctx, texts)
	vectorsProcessed.Add(float64(len(texts)))

	if s.flightClient != nil {
		for chunk := range ch {
			if chunk.Err != nil {
				log.Error().Err(chunk.Err).Msg("Inference error in stream")
				continue 
			}
			// Forward slice of original texts
			chunkTexts := texts[chunk.Offset : chunk.Offset+chunk.Count]
			if err := s.forwardToLongbow(ctx, chunkTexts, chunk.Vectors); err != nil {
				log.Error().Err(err).Msg("Error forwarding chunk to Longbow")
				// Keep going for other chunks? Or abort?
			}
		}
	} else {
		// Drain the channel if no client
		for range ch {}
	}

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}

func (s *Server) forwardToLongbow(ctx context.Context, texts []string, flatBatch []float32) error {
	curBatchSize := len(texts)
	
	// Text Column
	tb := s.sbPool.Get().(*array.StringBuilder)
	defer s.sbPool.Put(tb)
	tb.AppendValues(texts, nil)
	textArr := tb.NewArray()
	defer textArr.Release()

	// Embedding Column
	if len(flatBatch) == 0 {
		return nil
	}
	cols := len(flatBatch) / curBatchSize

	// Zero-copy Arrow Data construction
	// Note: In a production scenario, we'd manage the lifecycle of the tensor memory
	// but here we rely on the flatBatch being owned by this function until RecordBatch is sent.
	resultBuf := memory.NewBufferBytes(arrow.Float32Traits.CastToBytes(flatBatch))
	
	fslType := arrow.FixedSizeListOf(int32(cols), arrow.PrimitiveTypes.Float32)
	
	valuesData := array.NewData(arrow.PrimitiveTypes.Float32, curBatchSize*cols, []*memory.Buffer{nil, resultBuf}, nil, 0, 0)
	defer valuesData.Release()
	
	fslData := array.NewData(
		fslType,
		curBatchSize,
		[]*memory.Buffer{nil}, 
		[]arrow.ArrayData{valuesData},
		0,
		0,
	)
	defer fslData.Release()
	embeddingArr := array.NewFixedSizeListData(fslData)
	defer embeddingArr.Release()

	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "text", Type: arrow.BinaryTypes.String},
			{Name: "embedding", Type: fslType},
		},
		nil,
	)
	
	rb := array.NewRecordBatch(schema, []arrow.Array{textArr, embeddingArr}, int64(curBatchSize))
	defer rb.Release()
	
	return s.flightClient.DoPut(ctx, s.datasetName, rb)
}


func (s *Server) handleEncodeArrow(w http.ResponseWriter, r *http.Request) {
	ctx, span := tracer.Start(r.Context(), "handleEncodeArrow")
	defer span.End()

	start := time.Now()
	defer func() {
		requestDuration.Observe(time.Since(start).Seconds())
	}()

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	reader, err := ipc.NewReader(r.Body, ipc.WithAllocator(s.alloc))
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create IPC reader: %v", err), http.StatusBadRequest)
		return
	}
	defer reader.Release()

	totalProcessed := 0
	
	for reader.Next() {
		rec := reader.Record()
		if rec.NumCols() == 0 {
			continue
		}
		
		// Expect "text" column. Assume 0 is text for now or by name "text"
		col := rec.Column(0) 
		// If schema has names, check for "text"
		indices := rec.Schema().FieldIndices("text")
		if len(indices) > 0 {
			col = rec.Column(indices[0])
		}

		// Convert Arrow string column to []string for EmbedBatch
		strArr, ok := col.(*array.String)
		if !ok {
			// Try Binary?
			binArr, okBin := col.(*array.Binary)
			if okBin {
				texts := make([]string, binArr.Len())
				for i := 0; i < binArr.Len(); i++ {
					texts[i] = string(binArr.Value(i))
				}
				// Process binary strings
				s.processBatch(ctx, texts)
				totalProcessed += len(texts)
				continue
			}
			
			log.Warn().Msg("First column is not String/Binary array, skipping batch")
			continue
		}

		texts := make([]string, strArr.Len())
		for i := 0; i < strArr.Len(); i++ {
			texts[i] = strArr.Value(i)
		}
		
		weight := int64(len(texts))
		if err := s.sem.Acquire(ctx, weight); err != nil {
			log.Error().Err(err).Msg("Failed to acquire semaphore for arrow batch")
			// For stream, maybe we just break or wait?
			// Acquire is blocking unless ctx canceled.
			// Ideally we wait.
			break
		}
		s.processBatch(ctx, texts)
		s.sem.Release(weight)
		
		totalProcessed += len(texts)
	}

	if reader.Err() != nil {
		log.Error().Err(reader.Err()).Msg("Error reading Arrow stream")
		http.Error(w, "Stream error", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "Processed %d vectors", totalProcessed)
}

func (s *Server) processBatch(ctx context.Context, texts []string) {
	// Pipeline this batch
	ch := s.embedder.EmbedBatch(ctx, texts)
	vectorsProcessed.Add(float64(len(texts)))

	if s.flightClient != nil {
		for chunk := range ch {
			if chunk.Err != nil {
				log.Error().Err(chunk.Err).Msg("Inference error in stream")
				continue 
			}
			chunkTexts := texts[chunk.Offset : chunk.Offset+chunk.Count]
			if err := s.forwardToLongbow(ctx, chunkTexts, chunk.Vectors); err != nil {
				log.Error().Err(err).Msg("Error forwarding chunk to Longbow")
			}
		}
	} else {
		for range ch {}
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}
