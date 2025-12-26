package embeddings

import (
	"fmt"

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
func NewEmbedder(vocabPath, weightsPath string, useGPU bool) (*Embedder, error) {
	tok, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	config := model.DefaultBertTinyConfig()
	
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

	loader := weights.NewLoader(bert)
	if err := loader.LoadFromRawBinary(weightsPath); err != nil {
		return nil, fmt.Errorf("failed to load weights: %w", err)
	}

	return &Embedder{
		model:     bert,
		tokenizer: tok,
	}, nil
}

// EmbedBatch generates embeddings for a batch of texts.
func (e *Embedder) EmbedBatch(texts []string) [][]float32 {
	inputs := make([]int, 0, len(texts)*32) // heuristic cap
	lengths := make([]int, len(texts))
	
	for i, text := range texts {
		_, ids := e.tokenizer.Tokenize(text)
		
		// Add [CLS] and [SEP]
		// [CLS]
		inputs = append(inputs, 101)
		// Tokens
		inputs = append(inputs, ids...)
		// [SEP]
		inputs = append(inputs, 102)
		
		lengths[i] = len(ids) + 2
	}
	
	// Forward Pass
	outputMatrix := e.model.ForwardBatch(inputs, lengths)
	
	// Extract rows into result slice
	r, c := outputMatrix.Dims()
	if r != len(texts) {
		// Should not happen
		return nil
	}
	
	results := make([][]float32, r)
	
	// Data() returns nil if on GPU, so using ToHost() which handles both cases
	// device.Tensor now returns []float32 from ToHost/Data
	data := outputMatrix.ToHost()
	
	for i := 0; i < r; i++ {
		// Create row slice
		// We can share memory if we want zero-copy, but safety first for API
		row := make([]float32, c)
		copy(row, data[i*c : (i+1)*c])
		results[i] = row
	}
	
	return results
}
