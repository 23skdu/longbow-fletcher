package embeddings

import (
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
)

// Embedder provides high-level text embedding capabilities.
type Embedder struct {
	tokenizer *tokenizer.WordPieceTokenizer
	model     *model.BertModel
}

// NewEmbedder initializes the embedder with vocab and weights.
func NewEmbedder(vocabPath, weightsPath string, config model.BertConfig) (*Embedder, error) {
	tk, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		return nil, err
	}

	m := model.NewBertModel(config)
	loader := weights.NewLoader(m)
	
	// Optional: Only load if weightsPath is provided
	if weightsPath != "" {
		if err := loader.LoadFromRawBinary(weightsPath); err != nil {
			return nil, err
		}
	}

	return &Embedder{
		tokenizer: tk,
		model:     m,
	}, nil
}

// EmbedBatch generates embeddings for a batch of texts.
// This is significantly faster for multiple inputs as it uses batched matrix operations.
func (e *Embedder) EmbedBatch(texts []string) [][]float64 {
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
	
	results := make([][]float64, r)
	data := outputMatrix.RawMatrix().Data
	
	for i := 0; i < r; i++ {
		row := make([]float64, c)
		copy(row, data[i*c : (i+1)*c])
		results[i] = row
	}
	
	return results
}

// Embed generates an embedding for the given text.
// For multiple texts, use EmbedBatch for better performance.
func (e *Embedder) Embed(text string) []float64 {
	oneBatch := e.EmbedBatch([]string{text})
	if len(oneBatch) == 0 {
		return nil
	}
	return oneBatch[0]
}
