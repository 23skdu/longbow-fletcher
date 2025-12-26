package embeddings

import (
	"fmt"
	"os"

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
	const internalBatchSize = 128
	allResults := make([][]float32, 0, len(texts))

	for i := 0; i < len(texts); i += internalBatchSize {
		end := i + internalBatchSize
		if end > len(texts) {
			end = len(texts)
		}

		batchTexts := texts[i:end]
		inputs := make([]int, 0, len(batchTexts)*32)
		lengths := make([]int, len(batchTexts))

		for j, text := range batchTexts {
			_, ids := e.tokenizer.Tokenize(text)

			// Add [CLS] and [SEP]
			inputs = append(inputs, 101) // [CLS]
			inputs = append(inputs, ids...)
			inputs = append(inputs, 102) // [SEP]

			lengths[j] = len(ids) + 2
		}

		// Forward Pass for this batch
		outputMatrix := e.model.ForwardBatch(inputs, lengths)

		// Extract rows
		r, c := outputMatrix.Dims()
		if r != len(batchTexts) {
			continue
		}

		data := outputMatrix.ToHost()
		for j := 0; j < r; j++ {
			row := make([]float32, c)
			copy(row, data[j*c:(j+1)*c])
			allResults = append(allResults, row)
		}
		
		e.model.Backend.PutTensor(outputMatrix)
	}

	return allResults
}
