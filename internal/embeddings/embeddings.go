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
	model     *model.BertModel
	tokenizer *tokenizer.WordPieceTokenizer
}

// NewEmbedder creates a new embedder.
// vocabPath and weightsPath are paths to the vocab.txt and model weights binary.
func NewEmbedder(vocabPath, weightsPath string, useGPU bool, modelType string) (*Embedder, error) {
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
	
	var backend device.Backend
	if useGPU {
		// NewMetalBackend will fallback or panic if not supported
		// Ideally we check availability
		backend = device.NewMetalBackend() // Use FP32 Metal by default, user can call NewMetalBackendFP16 via other means if we exposed it
		// Or we could expose a config option for FP16
	} else {
		backend = device.NewCPUBackend()
	}

	bert := model.NewBertModelWithBackend(config, backend)

	if weightsPath != "" && weightsPath != "bert_tiny.bin" {
		loader := weights.NewLoader(bert)
		if err := loader.LoadFromRawBinary(weightsPath); err != nil {
			return nil, fmt.Errorf("failed to load weights: %w", err)
		}
	} else if weightsPath == "bert_tiny.bin" {
		// Check if file exists, if not, skip with warning
		if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
			fmt.Printf("Warning: default weights file %s not found, using random initialization\n", weightsPath)
		} else {
			loader := weights.NewLoader(bert)
			if err := loader.LoadFromRawBinary(weightsPath); err != nil {
				return nil, fmt.Errorf("failed to load weights: %w", err)
			}
		}
	}

	return &Embedder{
		model:     bert,
		tokenizer: tok,
	}, nil
}

// EmbedBatch generates embeddings for a batch of texts.
// It processes texts in internal batches of 128 to prevent VRAM exhaustion on GPUs.
func (e *Embedder) EmbedBatch(texts []string) [][]float32 {
	if len(texts) == 0 {
		return nil
	}

	// 1. Parallel Tokenization
	type tokenizedResult struct {
		ids []int
		len int
	}
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

	const internalBatchSize = 128
	allResults := make([][]float32, len(texts))
	
	// 2. Sequential GPU Processing (with parallel extraction)
	for i := 0; i < len(texts); i += internalBatchSize {
		batchEnd := i + internalBatchSize
		if batchEnd > len(texts) {
			batchEnd = len(texts)
		}

		batchCount := batchEnd - i
		totalTokens := 0
		for j := i; j < batchEnd; j++ {
			totalTokens += results[j].len
		}

		inputs := make([]int, 0, totalTokens)
		lengths := make([]int, batchCount)

		for j := 0; j < batchCount; j++ {
			res := results[i+j]
			inputs = append(inputs, 101) // [CLS]
			inputs = append(inputs, res.ids...)
			inputs = append(inputs, 102) // [SEP]
			lengths[j] = res.len
		}

		// Forward Pass for this batch
		outputMatrix := e.model.ForwardBatch(inputs, lengths)

		// 3. Parallel Extraction
		data := outputMatrix.ToHost()
		_, c := outputMatrix.Dims()
		
		extractWG := sync.WaitGroup{}
		extractWorkers := 4 // Small number for data extraction
		if batchCount < extractWorkers {
			extractWorkers = batchCount
		}
		
		eSize := (batchCount + extractWorkers - 1) / extractWorkers
		for w := 0; w < extractWorkers; w++ {
			eStart := w * eSize
			if eStart >= batchCount {
				break
			}
			eEnd := eStart + eSize
			if eEnd > batchCount {
				eEnd = batchCount
			}
			
			extractWG.Add(1)
			go func(s, end int) {
				defer extractWG.Done()
				for j := s; j < end; j++ {
					row := make([]float32, c)
					copy(row, data[j*c:(j+1)*c])
					allResults[i+j] = row
				}
			}(eStart, eEnd)
		}
		extractWG.Wait()
		
		e.model.Backend.PutTensor(outputMatrix)
	}

	return allResults
}
