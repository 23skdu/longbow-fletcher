package embeddings

import (
	"os"
	"testing"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
)

func createTempVocab(t *testing.T) string {
	f, err := os.CreateTemp("", "vocab")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Add some basic tokens
	tokens := []string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"hello", "world", "test", "sentence", "embedding",
		"##lo", "##ld",
	}
	for _, tok := range tokens {
		f.WriteString(tok + "\n")
	}
	return f.Name()
}

func TestNewEmbedder(t *testing.T) {
	vocabPath := createTempVocab(t)
	defer os.Remove(vocabPath)
	
	config := model.DefaultBertTinyConfig()
	
	// Test safe initialization
	e, err := NewEmbedder(vocabPath, "", config)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	if e == nil {
		t.Fatal("Embedder is nil")
	}
}

func TestEmbedder_Embed(t *testing.T) {
	vocabPath := createTempVocab(t)
	defer os.Remove(vocabPath)
	
	config := model.DefaultBertTinyConfig()
	e, err := NewEmbedder(vocabPath, "", config)
	if err != nil {
		t.Fatal(err)
	}
	
	tests := []string{
		"hello world", // lowercase to match vocab
		"test sentence",
		"", 
	}
	
	for _, text := range tests {
		vec := e.Embed(text)
		
		if len(vec) != config.HiddenSize {
			t.Errorf("Expected vector length %d, got %d for input '%s'", config.HiddenSize, len(vec), text)
		}
	}
}

func BenchmarkEmbedder_Embed(b *testing.B) {
	// Setup (ignore cleanup for bench or do it properly)
	f, _ := os.CreateTemp("", "bench_vocab")
	defer os.Remove(f.Name())
	f.WriteString("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n")
	f.Close()
	
	config := model.DefaultBertTinyConfig()
	e, _ := NewEmbedder(f.Name(), "", config)
	
	text := "hello world"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Embed(text)
	}
}
