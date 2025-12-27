package embeddings

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/device"
)

func createTempVocab(t *testing.T) string {
	f, err := os.CreateTemp("", "vocab")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()

	tokens := []string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"hello", "world", "test", "sentence", "embedding",
		"##lo", "##ld",
	}
	for _, tok := range tokens {
		_, _ = f.WriteString(tok + "\n")
	}
	return f.Name()
}

func TestEmbedder_EmbedBatch(t *testing.T) {
	vocabPath := createTempVocab(t)
	defer func() { _ = os.Remove(vocabPath) }()
	
	// Manually construct tokenizer
	tok, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		t.Fatal(err)
	}

	// Manually construct model (avoid weight loading)
	config := model.DefaultBertTinyConfig()
	backend := device.NewCPUBackend()
	bert := model.NewBertModelWithBackend(config, backend) 
	// initWeights is called in NewBertModelWithBackend, so it has random weights.

	e := &Embedder{
		model:     bert,
		tokenizer: tok,
	}
	
	tests := []string{
		"hello world", 
		"test sentence",
	}
	
	vectors := e.EmbedBatch(tests)
	
	if len(vectors) != len(tests) {
		t.Errorf("Expected %d vectors, got %d", len(tests), len(vectors))
	}
	
	for i, vec := range vectors {
		if len(vec) != config.HiddenSize {
			t.Errorf("Test %d: Expected vector length %d, got %d", i, config.HiddenSize, len(vec))
		}
		
		// Check for non-zero (highly unlikely to be all zero with Xavier init)
		hasNonZero := false
		for _, v := range vec {
			if v != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Errorf("Test %d: Vector is all zeros", i)
		}
	}
}
