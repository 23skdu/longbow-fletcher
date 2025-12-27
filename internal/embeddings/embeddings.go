package embeddings

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
	"github.com/23skdu/longbow-fletcher/internal/device"
)

// Embedder manages the tokenization and model inference.
type Embedder struct {
	models            []*model.BertModel
	tokenizer         *tokenizer.WordPieceTokenizer
	internalBatchSize int
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
		if runtime.GOOS == "darwin" {
			d := device.NewMetalBackend()
			deviceCount = d.DeviceCount()
		} else if runtime.GOOS == "linux" {
			d := device.NewCudaBackend()
			deviceCount = d.DeviceCount()
		}
	}

	fmt.Printf("Initializing Embedder with %d device(s), precision: %s\n", deviceCount, precision)
	models := make([]*model.BertModel, deviceCount)

	// Pre-load weights into memory ONCE to avoid disk I/O per device
	// var weightData []byte // TODO: Refactor Loader later
	
	for i := 0; i < deviceCount; i++ {
		var backend device.Backend
		if useGPU {
			if runtime.GOOS == "darwin" {
				if precision == "fp16" {
					backend = device.NewMetalBackendFP16()
				} else {
					backend = device.NewMetalBackend()
				}
			} else if runtime.GOOS == "linux" {
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
					fmt.Printf("Warning: default weights file %s not found, using random initialization\n", weightsPath)
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
		if runtime.GOOS == "darwin" {
			batchSize = 256
		} else if runtime.GOOS == "linux" {
			batchSize = 512
		}
	}

	return &Embedder{
		models:            models,
		tokenizer:         tok,
		internalBatchSize: batchSize,
	}, nil
}

// EmbedBatch generates embeddings for a batch of texts.
// It processes texts in internal batches of 128 to prevent VRAM exhaustion on GPUs.
func (e *Embedder) EmbedBatch(texts []string) [][]float32 {
	if len(texts) == 0 {
		return nil
	}

	// 1. Parallel Tokenization
	// Uses package-level tokenizedResult struct
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

	// 2. Multi-GPU Dispatch
	allResults := make([][]float32, len(texts))
	numDevices := len(e.models)
	
	// Split work across devices
	itemsPerDevice := (len(texts) + numDevices - 1) / numDevices
	
	var deviceWg sync.WaitGroup
	for d := 0; d < numDevices; d++ {
		start := d * itemsPerDevice
		if start >= len(texts) {
			break
		}
		end := start + itemsPerDevice
		if end > len(texts) {
			end = len(texts)
		}
		
		deviceWg.Add(1)
		go func(devId, s, eIdx int) {
			defer deviceWg.Done()
			m := e.models[devId]
			runInferenceOnDevice(m, e.internalBatchSize, results[s:eIdx], allResults, s)
		}(d, start, end)
	}
	deviceWg.Wait()

	return allResults
}

// tokenizedResult is a helper struct for passing data
type tokenizedResult struct {
	ids []int
	len int
}

func runInferenceOnDevice(m *model.BertModel, batchSize int, inputs []tokenizedResult, outputs [][]float32, globalOffset int) {
	// Double-buffered GPU Processing (per device)
	var prevOutput device.Tensor
	var prevBatchIdx int

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

		// While GPU handles current batch, CPU extracts results from PREVIOUS batch
		if prevOutput != nil {
			prevOutput.ExtractTo(outputs, globalOffset+prevBatchIdx)
			m.Backend.PutTensor(prevOutput)
		}

		prevOutput = currentOutput
		prevBatchIdx = i
	}

	// Handle the final batch
	if prevOutput != nil {
		prevOutput.ExtractTo(outputs, globalOffset+prevBatchIdx)
		m.Backend.PutTensor(prevOutput)
	}
}
