package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/float16"
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

	"github.com/23skdu/longbow-fletcher/internal/client"
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

	// Flight Metrics
	flightBytesSent = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fletcher_flight_bytes_sent_total",
		Help: "Total bytes sent via Arrow Flight",
	})
	flightRequests = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fletcher_flight_requests_total",
		Help: "Total Arrow Flight DoPut requests",
	})
	flightDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "fletcher_flight_duration_seconds",
		Help:    "Time spent in Flight DoPut",
		Buckets: prometheus.DefBuckets,
	})
)

type EmbedderInterface interface {
	EmbedBatch(ctx context.Context, texts []string) <-chan embeddings.StreamResult
	ProxyEmbedBatch(ctx context.Context, texts []string) []float32
	GetVRAMUsage() (allocated int64, total int64)
	EstimateVRAM(numSequences int, totalBytes int) int64
}

type FlightClientInterface interface {
	DoPut(ctx context.Context, datasetName string, record arrow.RecordBatch) error
	Close() error
	StartStream(ctx context.Context, datasetName string) *client.AsyncStream
}

type Server struct {
	embedder     EmbedderInterface
	flightClient FlightClientInterface
	stream       *client.AsyncStream
	datasetName  string
	alloc        memory.Allocator
	sbPool       sync.Pool
	sem          *semaphore.Weighted
	vramSem      *semaphore.Weighted
	TransportFmt string
	ModelType    string
}

func NewServer(embedder EmbedderInterface, fc FlightClientInterface, dataset string, maxConcurrent int, maxVRAM int64, transportFmt string, modelType string) *Server {
	var vs *semaphore.Weighted
	if maxVRAM > 0 {
		vs = semaphore.NewWeighted(maxVRAM)
	}

	s := &Server{
		embedder:     embedder,
		flightClient: fc,
		datasetName:  dataset,
		alloc:        memory.NewGoAllocator(),
		sbPool: sync.Pool{
			New: func() interface{} {
				return array.NewStringBuilder(memory.DefaultAllocator)
			},
		},
		sem:          semaphore.NewWeighted(int64(maxConcurrent)),
		vramSem:      vs,
		TransportFmt: transportFmt,
		ModelType:    modelType,
	}

	if fc != nil {
		s.stream = fc.StartStream(context.Background(), dataset)
	}
	return s
}

func startServer(addr string, embedder EmbedderInterface, fc FlightClientInterface, dataset string, maxConcurrent int, maxVRAM int64, transportFmt string, modelType string) {
	srv := NewServer(embedder, fc, dataset, maxConcurrent, maxVRAM, transportFmt, modelType)

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
	http.HandleFunc("/encode", srv.recoverMiddleware(srv.handleEncode))
	http.HandleFunc("/encode/arrow", srv.recoverMiddleware(srv.handleEncodeArrow))

	http.HandleFunc("/health", srv.handleHealth)
	http.HandleFunc("/healthz", srv.handleHealthz)
	http.HandleFunc("/readyz", srv.handleReadyz)

	http.HandleFunc("/v1/embeddings", srv.recoverMiddleware(srv.handleV1Embeddings))
	http.HandleFunc("/v1/embeddings/batch", srv.recoverMiddleware(srv.handleV1EmbeddingsBatch))
	http.HandleFunc("/v1/models", srv.handleV1Models)
	http.HandleFunc("/v1/models/list", srv.handleV1ModelsList)
	http.HandleFunc("/v1/rerank", srv.recoverMiddleware(srv.handleV1Rerank))
	http.HandleFunc("/healthz", srv.handleHealthz)
	http.HandleFunc("/readyz", srv.handleReadyz)

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

	// Enhanced logging with Trace ID and Client IP
	sc := span.SpanContext()
	logger := log.With().
		Str("client_ip", r.RemoteAddr).
		Str("user_agent", r.UserAgent()).
		Logger()

	if sc.HasTraceID() {
		logger = logger.With().Str("trace_id", sc.TraceID().String()).Logger()
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

	logger.Info().Int("sequence_count", len(texts)).Msg("Received encode request")

	span.SetAttributes(
		attribute.Int("sequence_count", len(texts)),
	)

	// Use total character count for estimation
	totalBytes := 0
	for _, t := range texts {
		totalBytes += len(t)
	}

	// Apply Nomic Prefix if needed
	if s.ModelType == "nomic-embed-text" || s.ModelType == "nomic-embed-text-v1.5" {
		taskType := r.URL.Query().Get("task")
		if taskType == "" {
			taskType = "search_document" // Default
		}

		prefix := ""
		switch taskType {
		case "search_query":
			prefix = "search_query: "
		case "search_document":
			prefix = "search_document: "
		default:
			prefix = taskType + ": "
		}

		if prefix != "" {
			for i := range texts {
				texts[i] = prefix + texts[i]
			}
		}
	}

	// Admission Control
	// 1. Sequence Count Limit
	weight := int64(len(texts))
	if err := s.sem.Acquire(ctx, weight); err != nil {
		logger.Error().Err(err).Msg("Failed to acquire semaphore")
		http.Error(w, "Server busy", http.StatusServiceUnavailable)
		return
	}
	defer s.sem.Release(weight)

	// 2. VRAM Limit
	if s.vramSem != nil {
		estVRAM := s.embedder.EstimateVRAM(len(texts), totalBytes)
		if err := s.vramSem.Acquire(ctx, estVRAM); err != nil {
			logger.Error().
				Err(err).
				Int64("requested_vram", estVRAM).
				Msg("Failed to acquire VRAM semaphore: insufficient capacity")
			http.Error(w, fmt.Sprintf("Server busy (VRAM): requested %d bytes", estVRAM), http.StatusServiceUnavailable)
			return
		}
		defer s.vramSem.Release(estVRAM)
	}

	// 2 & 3. Embed and Forward (Pipelined)
	embedCtx := ctx

	// Dataset ID for caching
	reqDataset := r.URL.Query().Get("dataset")
	if reqDataset == "" {
		reqDataset = s.datasetName
	}
	if reqDataset != "" {
		embedCtx = embeddings.WithDatasetID(embedCtx, reqDataset)
	}

	if s.TransportFmt == "fp16" {
		embedCtx = embeddings.WithOutputFormat(embedCtx, "fp16")
	}
	ch := s.embedder.EmbedBatch(embedCtx, texts)
	vectorsProcessed.Add(float64(len(texts)))

	if s.flightClient != nil {
		for chunk := range ch {
			if chunk.Err != nil {
				logger.Error().Err(chunk.Err).Msg("Inference error in stream")
				continue
			}
			// Forward slice of original texts
			chunkTexts := texts[chunk.Offset : chunk.Offset+chunk.Count]
			if err := s.forwardToLongbow(ctx, chunkTexts, chunk); err != nil {
				logger.Error().Err(err).Msg("Error forwarding chunk to Longbow")
				// Keep going for other chunks? Or abort?
			}
		}
	} else {
		// Drain the channel if no client
		for range ch {
		}
	}

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}

func (s *Server) forwardToLongbow(ctx context.Context, texts []string, res embeddings.StreamResult) error {
	curBatchSize := len(texts)

	// Text Column
	tb := s.sbPool.Get().(*array.StringBuilder)
	defer s.sbPool.Put(tb)
	tb.AppendValues(texts, nil)
	textArr := tb.NewArray()
	defer textArr.Release()

	// Embedding Column
	var embedBytes int
	var fslType *arrow.FixedSizeListType
	var valuesData *array.Data

	if res.RawBytes != nil && s.TransportFmt == "fp16" {
		// Zero-copy path for FP16
		resultBuf := memory.NewBufferBytes(res.RawBytes)
		dim := len(res.RawBytes) / (curBatchSize * 2)
		fslType = arrow.FixedSizeListOf(int32(dim), arrow.FixedWidthTypes.Float16)
		valuesData = array.NewData(arrow.FixedWidthTypes.Float16, curBatchSize*dim, []*memory.Buffer{nil, resultBuf}, nil, 0, 0)
		embedBytes = len(res.RawBytes)
	} else if len(res.Vectors) > 0 {
		// FP32 path (or fallback)
		flatBatch := res.Vectors
		cols := len(flatBatch) / curBatchSize

		if s.TransportFmt == "fp16" {
			// Convert FP32 -> FP16 manually
			fp16Data := make([]float16.Num, len(flatBatch))
			for i, v := range flatBatch {
				fp16Data[i] = float16.New(v)
			}
			byteData := make([]byte, len(flatBatch)*2)
			for i, n := range fp16Data {
				u := n.Uint16()
				byteData[i*2] = byte(u)
				byteData[i*2+1] = byte(u >> 8)
			}
			resultBuf := memory.NewBufferBytes(byteData)
			fslType = arrow.FixedSizeListOf(int32(cols), arrow.FixedWidthTypes.Float16)
			valuesData = array.NewData(arrow.FixedWidthTypes.Float16, curBatchSize*cols, []*memory.Buffer{nil, resultBuf}, nil, 0, 0)
			embedBytes = len(byteData)
		} else {
			// FP32
			resultBuf := memory.NewBufferBytes(arrow.Float32Traits.CastToBytes(flatBatch))
			fslType = arrow.FixedSizeListOf(int32(cols), arrow.PrimitiveTypes.Float32)
			valuesData = array.NewData(arrow.PrimitiveTypes.Float32, curBatchSize*cols, []*memory.Buffer{nil, resultBuf}, nil, 0, 0)
			embedBytes = len(flatBatch) * 4
		}
	} else {
		return nil // No data?
	}
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

	fslArr := array.NewFixedSizeListData(fslData)
	defer fslArr.Release()

	rec := array.NewRecord(
		arrow.NewSchema(
			[]arrow.Field{
				{Name: "text", Type: arrow.BinaryTypes.String},
				{Name: "embedding", Type: fslType},
			},
			nil,
		),
		[]arrow.Array{textArr, fslArr},
		int64(curBatchSize),
	)
	defer rec.Release()

	start := time.Now()
	// Use Async Stream
	if s.stream != nil {
		s.stream.Send(rec)
	} else {
		// Fallback (should not happen if fc is not nil)
		err := s.flightClient.DoPut(ctx, s.datasetName, rec)
		if err != nil {
			return err
		}
	}
	duration := time.Since(start).Seconds()

	flightRequests.Inc()
	flightDuration.Observe(duration)
	flightBytesSent.Add(float64(embedBytes)) // Estimate

	return nil
}

func (s *Server) handleEncodeArrow(w http.ResponseWriter, r *http.Request) {
	ctx, span := tracer.Start(r.Context(), "handleEncodeArrow")
	defer span.End()

	// Enhanced logging with Trace ID and Client IP
	sc := span.SpanContext()
	logger := log.With().
		Str("client_ip", r.RemoteAddr).
		Str("user_agent", r.UserAgent()).
		Logger()

	if sc.HasTraceID() {
		logger = logger.With().Str("trace_id", sc.TraceID().String()).Logger()
	}

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

			logger.Warn().Msg("First column is not String/Binary array, skipping batch")
			continue
		}

		texts := make([]string, strArr.Len())
		for i := 0; i < strArr.Len(); i++ {
			texts[i] = strArr.Value(i)
		}

		// Apply Nomic Prefix if needed (Arrow path)
		if s.ModelType == "nomic-embed-text" || s.ModelType == "nomic-embed-text-v1.5" {
			taskType := r.URL.Query().Get("task")
			if taskType == "" {
				taskType = "search_document" // Default
			}

			prefix := ""
			switch taskType {
			case "search_query":
				prefix = "search_query: "
			case "search_document":
				prefix = "search_document: "
			default:
				prefix = taskType + ": "
			}

			if prefix != "" {
				for i := range texts {
					texts[i] = prefix + texts[i]
				}
			}
		}

		// Admission Control setup
		totalBytes := 0
		for _, t := range texts {
			totalBytes += len(t)
		}

		weight := int64(len(texts))
		if err := s.sem.Acquire(ctx, weight); err != nil {
			logger.Error().Err(err).Msg("Failed to acquire semaphore for arrow batch")
			break
		}

		// VRAM Limit (nested to ensure we release both)
		var vramAcquired int64
		if s.vramSem != nil {
			estVRAM := s.embedder.EstimateVRAM(len(texts), totalBytes)
			if err := s.vramSem.Acquire(ctx, estVRAM); err != nil {
				logger.Error().Err(err).Msg("Failed to acquire VRAM semaphore for arrow batch")
				s.sem.Release(weight) // Back out
				break
			}
			vramAcquired = estVRAM
		}

		s.processBatch(ctx, texts)

		if s.vramSem != nil {
			s.vramSem.Release(vramAcquired)
		}
		s.sem.Release(weight)

		totalProcessed += len(texts)
	}

	if reader.Err() != nil {
		logger.Error().Err(reader.Err()).Msg("Error reading Arrow stream")
		http.Error(w, "Stream error", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "Processed %d vectors", totalProcessed)
}

func (s *Server) processBatch(ctx context.Context, texts []string) {
	// Pipeline this batch
	embedCtx := ctx
	if s.TransportFmt == "fp16" {
		embedCtx = embeddings.WithOutputFormat(ctx, "fp16")
	}
	ch := s.embedder.EmbedBatch(embedCtx, texts)
	vectorsProcessed.Add(float64(len(texts)))

	if s.flightClient != nil {
		for chunk := range ch {
			if chunk.Err != nil {
				log.Error().Err(chunk.Err).Msg("Inference error in stream")
				continue
			}
			chunkTexts := texts[chunk.Offset : chunk.Offset+chunk.Count]
			if err := s.forwardToLongbow(ctx, chunkTexts, chunk); err != nil {
				log.Error().Err(err).Msg("Error forwarding chunk to Longbow")
			}
		}
	} else {
		for range ch {
		}
	}
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}

func (s *Server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	// Check if models are loaded (simple verification)
	if s.embedder == nil {
		http.Error(w, "Not Ready", http.StatusServiceUnavailable)
		return
	}

	// Optional: Check VRAM availability or basic inference capability
	// For now, if we are up and have an embedder, we are ready.

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("Ready"))
}

func (s *Server) recoverMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				// Metrics
				embeddings.PanicTotal.Inc()

				log.Error().
					Interface("panic", err).
					Str("stack", string(debugStack())).
					Msg("Panic recovered in HTTP handler")

				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next(w, r)
	}
}

func debugStack() []byte {
	// Simple stack trace
	var buf [4096]byte
	n := runtime.Stack(buf[:], false)
	return buf[:n]
}

// Deprecated: use handleHealthz
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.handleHealthz(w, r)
}

type V1EmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type V1EmbeddingResponse struct {
	Object string        `json:"object"`
	Data   []V1Embedding `json:"data"`
	Model  string        `json:"model"`
	Usage  V1Usage       `json:"usage"`
}

type V1Embedding struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type V1Usage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

func (s *Server) handleV1Embeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req V1EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	embeddings := s.embedder.ProxyEmbedBatch(r.Context(), []string{req.Input})

	resp := V1EmbeddingResponse{
		Object: "list",
		Data: []V1Embedding{
			{
				Object:    "embedding",
				Embedding: embeddings,
				Index:     0,
			},
		},
		Model: req.Model,
		Usage: V1Usage{
			PromptTokens: len(req.Input),
			TotalTokens:  len(req.Input),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

type V1BatchEmbeddingRequest struct {
	Inputs []string `json:"inputs"`
	Model  string   `json:"model"`
}

func (s *Server) handleV1EmbeddingsBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req V1BatchEmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	embeddings := s.embedder.ProxyEmbedBatch(r.Context(), req.Inputs)

	dim := 384
	if len(embeddings) > 0 {
		dim = len(embeddings) / len(req.Inputs)
	}

	data := make([]V1Embedding, len(req.Inputs))
	for i := range req.Inputs {
		start := i * dim
		end := start + dim
		data[i] = V1Embedding{
			Object:    "embedding",
			Embedding: embeddings[start:end],
			Index:     i,
		}
	}

	resp := V1EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  req.Model,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

type V1Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	resp := V1Model{
		ID:      "fletcher-embed",
		Object:  "model",
		Created: 1700000000,
		OwnedBy: "fletcher",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleV1ModelsList(w http.ResponseWriter, r *http.Request) {
	resp := map[string]interface{}{
		"object": "list",
		"data": []V1Model{
			{
				ID:      "fletcher-embed",
				Object:  "model",
				Created: 1700000000,
				OwnedBy: "fletcher",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

type V1RerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	Model     string   `json:"model"`
	TopN      int      `json:"top_n"`
}

type V1RerankResponse struct {
	Model   string           `json:"model"`
	Results []V1RerankResult `json:"results"`
}

type V1RerankResult struct {
	Index    int     `json:"index"`
	Document string  `json:"document"`
	Score    float64 `json:"score"`
}

func (s *Server) handleV1Rerank(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req V1RerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.TopN == 0 {
		req.TopN = len(req.Documents)
	}

	queryEmb := s.embedder.ProxyEmbedBatch(r.Context(), []string{req.Query})

	type scoredDoc struct {
		index    int
		document string
		score    float64
	}

	var scored []scoredDoc
	for i, doc := range req.Documents {
		docEmb := s.embedder.ProxyEmbedBatch(r.Context(), []string{doc})
		score := cosineSimilarityFloat64(queryEmb, docEmb)
		scored = append(scored, scoredDoc{index: i, document: doc, score: score})
	}

	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[i].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	topN := req.TopN
	if topN > len(scored) {
		topN = len(scored)
	}

	results := make([]V1RerankResult, topN)
	for i := 0; i < topN; i++ {
		results[i] = V1RerankResult{
			Index:    scored[i].index,
			Document: scored[i].document,
			Score:    scored[i].score,
		}
	}

	resp := V1RerankResponse{
		Model:   req.Model,
		Results: results,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func cosineSimilarityFloat64(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	var dot, normA, normB float64
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	for i := 0; i < minLen; i++ {
		dot += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
