package embeddings

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync"
	"strings"
	"time"

	"crypto/sha256"
	"encoding/hex"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/cache"
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
	maxBatchTokens    int
	cache             cache.VectorCache
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
		// Wrap in recover to handle panic if GPU init fails
		func() {
			defer func() {
				if r := recover(); r != nil {
					log.Error().Msgf("GPU probe failed: %v. Falling back to CPU.", r)
					useGPU = false
				}
			}()
			
			switch runtime.GOOS {
			case "darwin":
				d := device.NewMetalBackend()
				deviceCount = d.DeviceCount()
			case "linux":
				d := device.NewCudaBackend()
				deviceCount = d.DeviceCount()
			}
		}()
	}

	log.Info().
		Int("devices", deviceCount).
		Str("precision", precision).
		Bool("gpu_enabled", useGPU).
		Msg("Initializing Embedder")
	models := make([]*model.BertModel, deviceCount)

	// Pre-load weights into memory ONCE to avoid disk I/O per device
	// var weightData []byte // TODO: Refactor Loader later
	
	for i := 0; i < deviceCount; i++ {
		var backend device.Backend
		
		if useGPU {
			// Try to create GPU backend, fallback to CPU on panic
			func() {
				defer func() {
					if r := recover(); r != nil {
						log.Error().Int("device", i).Msgf("Failed to initialize GPU backend: %v. Falling back to CPU.", r)
						backend = device.NewCPUBackend()
					}
				}()
				
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
				if backend != nil {
					backend.SetDevice(i) // Pin backend to specific device
				}
			}()
		}
		
		// If GPU init failed (backend still nil) or useGPU was false
		if backend == nil {
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
	maxTokens := 2048 // Conservative default
	if useGPU {
		// Dynamic batch sizing based on platform
		switch runtime.GOOS {
		case "darwin":
			batchSize = 256
			maxTokens = 16384 // ~16k tokens allows good utilization of M1/M2/M3
		case "linux":
			batchSize = 512
			maxTokens = 16384 // VRAM dependent, keeping safe default
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
		maxBatchTokens:    maxTokens,
		cache:             cache.NewMapCache(),
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



type ctxKeyDatasetID struct{}

// WithDatasetID attaches a dataset ID to the context for caching purposes.
func WithDatasetID(ctx context.Context, datasetID string) context.Context {
	return context.WithValue(ctx, ctxKeyDatasetID{}, datasetID)
}

func processOutput(m *model.BertModel, t device.Tensor, dim int, indices []int, format string, out chan<- StreamResult, cacheToAdd cache.VectorCache, keysToAdd []string) {
	// Extract all data from tensor (flat)
	count := len(indices)
	
	if format == "fp16" {
		t16 := t.Cast(device.Float16)
		bytes := t16.ExtractBytes()
		
		if t16 != t {
			m.Backend.PutTensor(t16)
		}
		
		// FP16 is 2 bytes per element
		rowBytes := dim * 2
		
		for k, originalIdx := range indices {
			start := k * rowBytes
			end := start + rowBytes
			
			// Copy for safety/independence
			row := make([]byte, rowBytes)
			copy(row, bytes[start:end])
			
			// Note: We don't cache RawBytes currently in VectorCache (it stores []float32)
			// Enhancing cache to support raw bytes is future work.
			// For now, only cache misses if not using raw mode? 
			// Or we decode to float32 to cache? That's expensive.
			// Let's Skip caching on FP16 path for now OR implement caching later.
			// Assuming cache stores float32, we can't put raw bytes easily.
			
			out <- StreamResult{
				Offset:      originalIdx,
				Count:       1,
				RawBytes:    row,
				RawDataType: device.Float16,
			}
		}
	} else {
		// FP32
		chunk := make([]float32, count*dim)
		t.ExtractToFlat(chunk, 0)
		
		for k, originalIdx := range indices {
			start := k * dim
			end := start + dim
			vec := chunk[start:end]
			
			// Send Result
			// Note: vec is a slice of chunk. StreamResult usually wants independent ownership?
			// But for channel sending, if receiver copies, it's fine.
			// ProxyEmbedBatch copies.
			// If we cache it, we MUST copy.
			
			if cacheToAdd != nil && k < len(keysToAdd) {
				cacheToAdd.Put(keysToAdd[k], vec)
			}
			
			out <- StreamResult{
				Offset:  originalIdx,
				Count:   1,
				Vectors: vec,
			}
		}
	}
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

	// 1. Parallel Tokenization with Work Queue Distribution
	// Uses package-level tokenizedResult struct
	batchStart := time.Now()
	_, tSpan := tracer.Start(ctx, "Tokenization")
	results := make([]tokenizedResult, len(texts))
	
	// Use work queue for better load balancing
	// This handles imbalanced text lengths better than static chunks
	workQueue := make(chan int, len(texts))
	for i := range texts {
		workQueue <- i
	}
	close(workQueue)
	
	numWorkers := runtime.NumCPU()
	if numWorkers > 16 {
		numWorkers = 16 // Cap at 16 workers for tokenization
	}
	
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	
	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			for idx := range workQueue {
				_, ids := e.tokenizer.Tokenize(texts[idx])
				results[idx] = tokenizedResult{
					ids: ids,
					len: len(ids) + 2, // [CLS] + [SEP]
					originalIdx: idx,
				}
			}
		}()
	}
	
	wg.Wait()
	tSpan.End()
	
	// Export tokenization metrics
	tokenizationElapsed := time.Since(batchStart)
	totalTokens := 0
	for _, res := range results {
		totalTokens += res.len
	}
	tokenizationDuration.Observe(tokenizationElapsed.Seconds())
	if tokenizationElapsed.Seconds() > 0 {
		tokensPerSecond.Set(float64(totalTokens) / tokenizationElapsed.Seconds())
	}

	// 2. Multi-GPU Dispatch with Dynamic Load Balancing
	dim := e.models[0].Config.HiddenSize
	numDevices := len(e.models)
	
	// Check Cache (on Main Thread)
	datasetID, _ := ctx.Value(ctxKeyDatasetID{}).(string)
	
	var missInputs []tokenizedResult
	var cacheKeys []string
	
	if datasetID != "" && e.cache != nil {
		missInputs = make([]tokenizedResult, 0, len(results))
		cacheKeys = make([]string, 0, len(results)) // Aligned with missInputs
		
		for _, res := range results {
			// Compute hash key: SHA256(DatasetID + Text)
			// access text via tokenizer? No, we lost original text in results struct.
			// Using offset is tricky. 
			// We MUST check cache BEFORE tokenization to save tokenization?
			// Or check using text during tokenization? 
			// Tokenization is fast. Let's do it after tokenization using original text?
			// The `results` struct doesn't have text. `texts` slice does.
			
			// Let's use `texts[res.originalIdx]`
			text := texts[res.originalIdx]
			
			h := sha256.New()
			h.Write([]byte(datasetID))
			h.Write([]byte(text))
			key := hex.EncodeToString(h.Sum(nil))
			
			if vec, found := e.cache.Get(key); found {
				cacheHits.Inc()
				out <- StreamResult{
					Offset:  res.originalIdx,
					Count:   1,
					Vectors: vec,
				}
			} else {
				cacheMisses.Inc()
				missInputs = append(missInputs, res)
				cacheKeys = append(cacheKeys, key)
			}
		}
	} else {
		missInputs = results
	}
	
	totalTokens = 0
	for _, res := range missInputs {
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
		
		// Export weight to Prometheus
		deviceLabel := fmt.Sprintf("%d", i)
		gpuWeight.WithLabelValues(deviceLabel).Set(deviceWeights[i])
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
			
			if endIndex > len(missInputs) {
				endIndex = len(missInputs)
			}
			
			batchInputs := missInputs[startIndex:endIndex]
			batchKeys := []string(nil)
			if len(cacheKeys) > 0 {
				batchKeys = cacheKeys[startIndex:endIndex]
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
				
				// runInference expects explicit indicesMap now? No, we refactor runInference to accept originalIdx in Result.
				// But we need to pass cache keys to populate cache.
				runInferenceOnDevice(m, e.internalBatchSize, e.maxBatchTokens, batchInputs, dim, format, out, &e.gpuMetrics[devId], e.cache, batchKeys)
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
	
	// 3. Calculate Memory Usage per Sequence (Heuristic)
	// Check Backend Precision
	bytesPerElement := 4
	if strings.Contains(e.models[0].Backend.Name(), "FP16") {
		bytesPerElement = 2
	}
	
	// Dynamic Batching Cap: We never process more than maxBatchTokens at once per device
	if avgTokens*effectiveBatchSize > e.maxBatchTokens {
		// Effective concurrent tokens is capped
		effectiveBatchSize = e.maxBatchTokens / int(avgTokens)
		if effectiveBatchSize < 1 {
			effectiveBatchSize = 1
		}
		// Adjust for double buffering
		effectiveBatchSize *= 2
	}

	hiddenSize = e.models[0].Config.HiddenSize
	layers = e.models[0].Config.NumHiddenLayers
	
	// Heuristic Factors (Tunable):
	// Fixed overhead per batch
	const fixedOverhead = 10 * 1024 * 1024 // 10MB base overhead
	
	// Linear cost per token (Activations)
	// Estimate: Bytes * Layers * (10 * Hidden)
	linearFactor := int64(layers * 10 * hiddenSize * bytesPerElement)
	
	// Quadratic cost (Attention Matrix): Layers * Heads * SeqLen
	// (Only significant for long sequences)
	quadraticFactor := int64(layers * e.models[0].Config.NumAttentionHeads * bytesPerElement)

	// Per Sequence Cost
	seqLen := int64(avgTokens)
	costPerSeq := (seqLen * linearFactor) + (seqLen * seqLen * quadraticFactor)
	
	totalEst := fixedOverhead + (int64(effectiveBatchSize) * costPerSeq)
	
	// Add Safety Margin (20%)
	totalEst = int64(float64(totalEst) * 1.2)
	
	return totalEst
}
	


// tokenizedResult is a helper struct for passing data
type tokenizedResult struct {
	ids []int
	len int
	originalIdx int
}

func runInferenceOnDevice(m *model.BertModel, maxBatchSize int, maxTokens int, inputs []tokenizedResult, dim int, format string, out chan<- StreamResult, metrics *GPUMetrics, reportCache cache.VectorCache, cacheKeys []string) {
	// Track overall batch performance
	batchStart := time.Now()
	totalSequences := len(inputs)
	totalTokensProcessed := 0
	
	// Double-buffered GPU Processing (per device)
	var prevOutput device.Tensor
	var prevIndices []int
	var prevKeys []string

	count := len(inputs)
	i := 0
	for i < count {
		// Dynamic Batching: Fill batch until maxBatchSize OR maxTokens reached
		currentBatchTokens := 0
		currentBatchSize := 0
		
		for j := i; j < count; j++ {
			seqLen := inputs[j].len
			
			// Check limits
			if currentBatchSize >= maxBatchSize {
				break
			}
			if currentBatchTokens+seqLen > maxTokens && currentBatchSize > 0 {
				break
			}
			
			currentBatchTokens += seqLen
			currentBatchSize++
		}
		
		if currentBatchSize == 0 {
			// Should not happen unless maxTokens < single sequence length
			currentBatchSize = 1
			currentBatchTokens = inputs[i].len
		}
		
		batchEndIdx := i + currentBatchSize
		
		// Prepare inputs
		flatInputs := make([]int, 0, currentBatchTokens)
		lengths := make([]int, currentBatchSize)
		batchIndices := make([]int, currentBatchSize)
		batchKeysSubset := []string(nil)
		if len(cacheKeys) > 0 {
			batchKeysSubset = cacheKeys[i:batchEndIdx]
		}

		for k := 0; k < currentBatchSize; k++ {
			res := inputs[i+k]
			flatInputs = append(flatInputs, 101) // [CLS]
			flatInputs = append(flatInputs, res.ids...)
			flatInputs = append(flatInputs, 102) // [SEP]
			lengths[k] = res.len
			batchIndices[k] = res.originalIdx
		}
		
		totalTokensProcessed += currentBatchTokens

		// Start GPU for current batch
		currentOutput := m.ForwardBatch(flatInputs, lengths)

		// Async handle previous batch
		if prevOutput != nil {
			processOutput(m, prevOutput, dim, prevIndices, format, out, reportCache, prevKeys)
			m.Backend.PutTensor(prevOutput)
		}

		prevOutput = currentOutput
		prevIndices = batchIndices
		prevKeys = batchKeysSubset
		
		// Advance
		i = batchEndIdx
	}

	// Handle the final batch
	if prevOutput != nil {
		processOutput(m, prevOutput, dim, prevIndices, format, out, reportCache, prevKeys)
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
	
	// Export to Prometheus
	deviceLabel := fmt.Sprintf("%d", metrics.DeviceID)
	gpuThroughput.WithLabelValues(deviceLabel).Set(metrics.AvgThroughput)
	gpuBatchTime.WithLabelValues(deviceLabel).Set(elapsed.Seconds())
	gpuBatchCount.WithLabelValues(deviceLabel).Inc()
	gpuSequencesProcessed.WithLabelValues(deviceLabel).Add(float64(totalSequences))
	gpuTokensProcessed.WithLabelValues(deviceLabel).Add(float64(totalTokensProcessed))
	
	metrics.mu.Unlock()
}
