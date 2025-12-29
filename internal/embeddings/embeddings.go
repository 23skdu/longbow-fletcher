package embeddings

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/rs/zerolog/log"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

// GPUMetrics tracks performance metrics for a single GPU device
type GPUMetrics struct {
	DeviceID       int
	LastBatchTime  time.Duration
	BatchCount     int64
	TotalSequences int64
	TotalTokens    int64
	AvgThroughput  float64 // sequences/second
	mu             sync.Mutex
}

// Embedder manages the tokenization and model inference.
type Embedder struct {
	models            []*model.BertModel
	tokenizer         *tokenizer.WordPieceTokenizer
	internalBatchSize int
	gpuMetrics        []GPUMetrics
	metricsMu         sync.RWMutex
}

// NewEmbedder creates a new embedder.
// precision can be "fp32" (default) or "fp16".
func NewEmbedder(vocabPath, weightsPath string, useGPU bool, modelType string, precision string) (*Embedder, error) {
	tok, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	var config model.BertConfig
	switch modelType {
	case "bert-tiny":
		config = model.DefaultBertTinyConfig()
	case "nomic-embed-text":
		config = model.DefaultNomicConfig()
	default:
		return nil, fmt.Errorf("unknown model type: %s", modelType)
	}
	
	// Default to FP32
	if precision == "" {
		precision = "fp32"
	}
	
	deviceCount := 1
	if useGPU {
		// Probe for device count using a temporary backend
	switch runtime.GOOS {
	case "darwin":
		d := device.NewMetalBackend()
		deviceCount = d.DeviceCount()
	case "linux":
		d := device.NewCudaBackend()
		deviceCount = d.DeviceCount()
	}
	}

	log.Info().
		Int("devices", deviceCount).
		Str("precision", precision).
		Msg("Initializing Embedder")
	models := make([]*model.BertModel, deviceCount)

	// Pre-load weights into memory ONCE to avoid disk I/O per device
	// var weightData []byte // TODO: Refactor Loader later
	
	for i := 0; i < deviceCount; i++ {
		var backend device.Backend
		if useGPU {
			switch runtime.GOOS {
		case "darwin":
			if precision == "fp16" {
				backend = device.NewMetalBackendFP16()
			} else {
				backend = device.NewMetalBackend()
			}
		case "linux":
			if precision == "fp16" {
				backend = device.NewCudaBackendFP16()
			} else {
				backend = device.NewCudaBackend()
			}
		}
		backend.SetDevice(i) // Pin backend to specific device
		} else {
			backend = device.NewCPUBackend()
		}

		bert := model.NewBertModelWithBackend(config, backend)

		if weightsPath != "" && weightsPath != "bert_tiny.bin" {
			loader := weights.NewLoader(bert)
			if err := loader.LoadFromRawBinary(weightsPath); err != nil {
				return nil, fmt.Errorf("failed to load weights for device %d: %w", i, err)
			}
		} else if weightsPath == "bert_tiny.bin" {
			if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
				if i == 0 {
					log.Warn().Str("file", weightsPath).Msg("Default weights file not found, using random initialization")
				}
			} else {
				loader := weights.NewLoader(bert)
				if err := loader.LoadFromRawBinary(weightsPath); err != nil {
					return nil, fmt.Errorf("failed to load weights for device %d: %w", i, err)
				}
			}
		}
		models[i] = bert
	}

	// Default batch sizes based on device
	batchSize := 32
	if useGPU {
		// Dynamic batch sizing based on platform
		switch runtime.GOOS {
		case "darwin":
			batchSize = 256
		case "linux":
			batchSize = 512
		}
	}

	// Initialize GPU metrics
	metrics := make([]GPUMetrics, deviceCount)
	for i := 0; i < deviceCount; i++ {
		metrics[i] = GPUMetrics{
			DeviceID:      i,
			AvgThroughput: 1.0, // Default throughput for cold start
		}
	}

	return &Embedder{
		models:            models,
		tokenizer:         tok,
		internalBatchSize: batchSize,
		gpuMetrics:        metrics,
	}, nil
}

var tracer = otel.Tracer("fletcher-embeddings")

// StreamResult contains a partial result of an EmbedBatch call.
type StreamResult struct {
	Vectors     []float32
	RawBytes    []byte          // Zero-copy bytes from device (if requested)
	RawDataType device.DataType // Type of RawBytes
	Offset      int             // Position in the original request
	Count       int             // Number of sequences in this chunk
	Err         error
}

type ctxKeyOutputFormat struct{}

// WithOutputFormat returns a context with the desired output format hint.
// format: "fp16" or "fp32" (default).
func WithOutputFormat(ctx context.Context, format string) context.Context {
	return context.WithValue(ctx, ctxKeyOutputFormat{}, format)
}



func processOutput(m *model.BertModel, t device.Tensor, dim int, offset, count int, format string, out chan<- StreamResult) {
	// Helper to handle extraction and sending
	res := StreamResult{
		Offset: offset,
		Count:  count,
	}

	if format == "fp16" {
		// We want FP16 bytes.
		// If tensor is already FP16 (from backend), extract bytes.
		// If tensor is FP32, cast to FP16 then extract.
		
		// We can check t.DType if we exposed it on Tensor interface?
		// We exposed ExtractBytes.
		// `Cast` handles conversion.
		
		t16 := t.Cast(device.Float16) // Returns new tensor or same if already FP16
		// If Cast returns new tensor, we must free it (if it owns memory).
		// Tensor interface doesn't expose Free (Backends handle it or Finalizers).
		// Our Cast implementation returns a Tensor with Finalizer or Pool management.
		
		res.RawBytes = t16.ExtractBytes()
		res.RawDataType = device.Float16
		
		// If t16 was a new tensor copy, we should signal we are done with it?
		// PutTensor might handle returning to pool if it's a pooled tensor.
		// Cast implementation returned a manual alloc in some cases.
		// If we use PutTensor on it, the backend needs to know how to handle it.
		// MetalBackend.PutTensor checks ownsBuffer. 
		// So safe to PutTensor(t16) if different from t.
		
		if t16 != t {
			m.Backend.PutTensor(t16)
		}
		
	} else {
		// Default FP32 Vectors
		chunk := make([]float32, count*dim)
		t.ExtractToFlat(chunk, 0)
		res.Vectors = chunk
	}
	
	out <- res
}


// EmbedBatch generates embeddings for a batch of texts.
// It returns a channel that yields StreamResults as they become available.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) <-chan StreamResult {
	out := make(chan StreamResult, len(e.models)*2)
	ctx, span := tracer.Start(ctx, "EmbedBatch")
	defer span.End()

	if len(texts) == 0 {
		close(out)
		return out
	}

	span.SetAttributes(attribute.Int("sequence_count", len(texts)))

	// 1. Parallel Tokenization
	// Uses package-level tokenizedResult struct
	_, tSpan := tracer.Start(ctx, "Tokenization")
	results := make([]tokenizedResult, len(texts))
	numWorkers := runtime.NumCPU()
	if numWorkers > 16 {
		numWorkers = 16 // Cap at 16 workers for tokenization
	}
	
	var wg sync.WaitGroup
	chunkSize := (len(texts) + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		if start >= len(texts) {
			break
		}
		end := start + chunkSize
		if end > len(texts) {
			end = len(texts)
		}
		
		wg.Add(1)
		go func(s, eIdx int) {
			defer wg.Done()
			for i := s; i < eIdx; i++ {
				_, ids := e.tokenizer.Tokenize(texts[i])
				results[i] = tokenizedResult{
					ids: ids,
					len: len(ids) + 2, // [CLS] + [SEP]
				}
			}
		}(start, end)
	}
	wg.Wait()
	tSpan.End()

	// 2. Multi-GPU Dispatch with Dynamic Load Balancing
	dim := e.models[0].Config.HiddenSize
	numDevices := len(e.models)
	
	// Calculate total tokens for balancing
	totalTokens := 0
	for _, res := range results {
		totalTokens += res.len
	}
	
	// Calculate device weights based on historical performance
	deviceWeights := make([]float64, numDevices)
	totalWeight := 0.0
	
	e.metricsMu.RLock()
	for i := 0; i < numDevices; i++ {
		// Use average throughput as weight (sequences/second)
		deviceWeights[i] = e.gpuMetrics[i].AvgThroughput
		totalWeight += deviceWeights[i]
	}
	e.metricsMu.RUnlock()
	
	// Calculate target tokens per device based on weights
	targetTokensPerDevice := make([]int, numDevices)
	for i := 0; i < numDevices; i++ {
		if totalWeight > 0 {
			targetTokensPerDevice[i] = int(float64(totalTokens) * deviceWeights[i] / totalWeight)
		} else {
			// Fallback to equal distribution if no metrics yet
			targetTokensPerDevice[i] = totalTokens / numDevices
		}
		if targetTokensPerDevice[i] == 0 {
			targetTokensPerDevice[i] = 1
		}
	}

	go func() {
		defer close(out)
		_, iSpan := tracer.Start(ctx, "Inference")
		defer iSpan.End()

		var deviceWg sync.WaitGroup
		
		startIndex := 0
		for d := 0; d < numDevices; d++ {
			if startIndex >= len(results) {
				break
			}
			
			endIndex := startIndex
			currentTokens := 0
			targetTokens := targetTokensPerDevice[d]
			
			// Greedily accumulate items until we reach target load
			// But ensure at least one item if available
			for endIndex < len(results) {
				// Always take at least one item if we haven't taken any for this device yet
				if endIndex == startIndex {
					currentTokens += results[endIndex].len
					endIndex++
					continue
				}
				
				// Check if adding next item exceeds target significantly?
				// Simple approach: stop if we are above target, unless we are the last device
				if d < numDevices-1 && currentTokens >= targetTokens {
					break
				}
				
				currentTokens += results[endIndex].len
				endIndex++
			}
			
			// If last device, take everything remaining
			if d == numDevices-1 {
				endIndex = len(results)
			}
			
			if endIndex > len(results) {
				endIndex = len(results)
			}
			
			
			format, _ := ctx.Value(ctxKeyOutputFormat{}).(string)

			deviceWg.Add(1)
			go func(devId, s, eIdx int) {
				defer deviceWg.Done()
				log.Info().
					Int("device", devId).
					Int("sequences", eIdx-s).
					Int("tokens", currentTokens).
					Float64("weight", deviceWeights[devId]).
					Float64("throughput", e.gpuMetrics[devId].AvgThroughput).
					Msg("Dispatching batch")
				m := e.models[devId]
				runInferenceOnDevice(m, e.internalBatchSize, results[s:eIdx], s, dim, format, out, &e.gpuMetrics[devId])
			}(d, startIndex, endIndex)
			
			startIndex = endIndex
		}
		deviceWg.Wait()
	}()

	return out
}
// ProxyEmbedBatch is a helper for legacy code that wants the full slice.
func (e *Embedder) ProxyEmbedBatch(ctx context.Context, texts []string) []float32 {
	dim := e.models[0].Config.HiddenSize
	res := make([]float32, len(texts)*dim)
	ch := e.EmbedBatch(ctx, texts)
	for chunk := range ch {
		if chunk.Err != nil {
			continue 
		}
		copy(res[chunk.Offset*dim:], chunk.Vectors)
	}
	return res
}

// GetVRAMUsage returns the total VRAM usage across all devices.
func (e *Embedder) GetVRAMUsage() (allocated int64, total int64) {
	for _, m := range e.models {
		a, t := m.Backend.GetVRAMUsage()
		allocated += a
		// Total VRAM might be counted multiple times if multiple models share device?
		// But here we have 1 model per device (usually).
		// If multiple devices, summing totals is correct (Global VRAM).
		total += t
	}
	return
}

// EstimateVRAM returns the estimated VRAM usage in bytes for a given request.
// It assumes the request will be processed in chunks of e.internalBatchSize.
// numSequences: number of text strings in the request
// totalBytes: total number of bytes in the text strings (used to estimate token count)
func (e *Embedder) EstimateVRAM(numSequences int, totalBytes int) int64 {
	if numSequences == 0 {
		return 0
	}

	// 1. Estimate average tokens per sequence
	// English text is roughly 4 chars per token. Conservatively use 3.
	avgChars := totalBytes / numSequences
	avgTokens := avgChars / 3
	if avgTokens < 16 {
		avgTokens = 16 // Minimum floor
	}
	// Cap at model max position embeddings (e.g., 512 for BERT Tiny)
	maxPos := e.models[0].Config.MaxPositionEmbeddings
	if avgTokens > maxPos {
		avgTokens = maxPos
	}

	// 2. Determine effective batch size (capped by internal batch size)
	// We use 2x internal batch size because of double buffering in runInferenceOnDevice
	effectiveBatchSize := numSequences
	doubleBufferedCap := e.internalBatchSize * 2
	if effectiveBatchSize > doubleBufferedCap {
		effectiveBatchSize = doubleBufferedCap
	}
	
	// 3. Calculate Memory Usage per Sequence (Heuristic)
	// Based on BERT architecture:
	// Activation Memory ~= Layers * (Attention + MLP)
	// Attention: O(L^2) scores + O(L*D) projections
	// MLP: O(L*D*4)
	
	hiddenSize := e.models[0].Config.HiddenSize // e.g., 128
	layers := e.models[0].Config.NumHiddenLayers // e.g., 2
	
	// Float32 = 4 bytes
	bytesPerElement := 4
	
	// Per Layer:
	// Q, K, V, Out: 4 * (SeqLen * Hidden)
	// Attention Scores: (SeqLen * SeqLen) * Heads (optimization: reused?)
	// Let's assume naive scores allocation: SeqLen^2
	// Intermediates (MLP): SeqLen * IntermediateSize (usually 4*Hidden)
	
	// Simplified per-token cost (activations):
	// roughly: Layers * (4*Hidden + Intermediate + Hidden) * Bytes
	// + Attention Scores: Layers * Heads * SeqLen * Bytes (if linear attention) or SeqLen^2
	
	// For BERT Tiny (H=128, I=512, L=2):
	// Per Token: 2 * (512 + 512 + 128) * 4 ~= 10KB ?
	// Let's use a simpler linear constant derived from empirics or conservative bounds.
	// A standard BERT-base (Seq 512) takes ~1GB for batch 32? -> ~30MB/seq.
	// BERT Tiny is much smaller (1/20th parameters, smaller activations).
	
	// Heuristic Factors (Tunable):
	// Fixed overhead per batch
	const fixedOverhead = 10 * 1024 * 1024 // 10MB base overhead
	
	// Linear cost per token (Activations)
	// H=128, L=2 -> Small.
	// H=768, L=12 -> Large.
	// Estimate: 4 bytes * Layers * (10 * Hidden)
	linearFactor := int64(layers * 10 * hiddenSize * bytesPerElement)
	
	// Quadratic cost (Attention Matrix): Layers * Heads * SeqLen
	// (Only significant for long sequences)
	quadraticFactor := int64(layers * e.models[0].Config.NumAttentionHeads * bytesPerElement)

	// Per Sequence Cost
	seqLen := int64(avgTokens)
	costPerSeq := (seqLen * linearFactor) + (seqLen * seqLen * quadraticFactor)
	
	totalEst := fixedOverhead + (int64(effectiveBatchSize) * costPerSeq)
	
	return totalEst
}

// tokenizedResult is a helper struct for passing data
type tokenizedResult struct {
	ids []int
	len int
}

func runInferenceOnDevice(m *model.BertModel, batchSize int, inputs []tokenizedResult, baseOffset int, dim int, format string, out chan<- StreamResult, metrics *GPUMetrics) {
	// Track overall batch performance
	batchStart := time.Now()
	totalSequences := len(inputs)
	totalTokensProcessed := 0
	
	// Double-buffered GPU Processing (per device)
	var prevOutput device.Tensor
	var prevBatchCount int
	var prevOffset int

	count := len(inputs)
	for i := 0; i < count; i += batchSize {
		batchEnd := i + batchSize
		if batchEnd > count {
			batchEnd = count
		}

		batchCount := batchEnd - i
		totalTokens := 0
		for j := i; j < batchEnd; j++ {
			totalTokens += inputs[j].len
		}
		totalTokensProcessed += totalTokens

		flatInputs := make([]int, 0, totalTokens)
		lengths := make([]int, batchCount)

		for j := 0; j < batchCount; j++ {
			res := inputs[i+j]
			flatInputs = append(flatInputs, 101) // [CLS]
			flatInputs = append(flatInputs, res.ids...)
			flatInputs = append(flatInputs, 102) // [SEP]
			lengths[j] = res.len
		}

		// Start GPU for current batch
		currentOutput := m.ForwardBatch(flatInputs, lengths)

		// Async handle previous batch
		if prevOutput != nil {
			processOutput(m, prevOutput, dim, prevOffset, prevBatchCount, format, out)
			m.Backend.PutTensor(prevOutput)
		}

		prevOutput = currentOutput
		prevBatchCount = batchCount
		prevOffset = baseOffset + i
	}

	// Handle the final batch
	if prevOutput != nil {
		processOutput(m, prevOutput, dim, prevOffset, prevBatchCount, format, out)
		m.Backend.PutTensor(prevOutput)
	}
	
	// Update metrics after batch completion
	elapsed := time.Since(batchStart)
	
	metrics.mu.Lock()
	metrics.LastBatchTime = elapsed
	metrics.BatchCount++
	metrics.TotalSequences += int64(totalSequences)
	metrics.TotalTokens += int64(totalTokensProcessed)
	
	// Exponential moving average for throughput (alpha = 0.3 for responsiveness)
	currentThroughput := float64(totalSequences) / elapsed.Seconds()
	if metrics.AvgThroughput == 1.0 {
		// First real measurement, replace default
		metrics.AvgThroughput = currentThroughput
	} else {
		alpha := 0.3
		metrics.AvgThroughput = alpha*currentThroughput + (1-alpha)*metrics.AvgThroughput
	}
	metrics.mu.Unlock()
}
