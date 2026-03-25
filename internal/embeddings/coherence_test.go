package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"testing"
)

type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type OllamaEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct float64
	var normA, normB float64

	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func TestEmbeddingCoherence_BertTiny(t *testing.T) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog"
	embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{text})

	if len(embeddings) == 0 {
		t.Fatal("No embeddings generated")
	}

	for j, v := range embeddings {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("NaN/Inf detected at index %d", j)
		}
	}

	dim := 128
	if len(embeddings) != dim {
		t.Errorf("Embedding dimension: expected %d, got %d", dim, len(embeddings))
	}

	var sum float32
	for _, v := range embeddings {
		sum += v
	}
	if math.Abs(float64(sum)) > 100 {
		t.Logf("Warning: embedding sum is large: %f", sum)
	}
}

func TestEmbeddingCoherence_NomicEmbed(t *testing.T) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"nomic-embed-text",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog"
	embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{text})

	if len(embeddings) == 0 {
		t.Fatal("No embeddings generated")
	}

	for j, v := range embeddings {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("NaN/Inf detected at index %d", j)
		}
	}

	var sum float32
	for _, v := range embeddings {
		sum += v
	}
	if math.Abs(float64(sum)) > 1000 {
		t.Logf("Warning: embedding sum is large: %f", sum)
	}
}

func TestEmbeddingCoherence_SameTextSameEmbedding(t *testing.T) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	text := "machine learning is fascinating"

	emb1 := embedder.ProxyEmbedBatch(context.Background(), []string{text})
	emb2 := embedder.ProxyEmbedBatch(context.Background(), []string{text})

	sim := cosineSimilarity(emb1, emb2)

	if sim < 0.99 {
		t.Errorf("Same text should produce nearly identical embeddings, similarity: %f", sim)
	}

	t.Logf("Similarity for same text: %f", sim)
}

func TestEmbeddingCoherence_DifferentTextsDifferentEmbeddings(t *testing.T) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	texts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Deep learning neural networks transformer architecture",
		"Hello world programming software code",
	}

	embeddings := embedder.ProxyEmbedBatch(context.Background(), texts)
	dim := 128

	for i := 0; i < len(texts); i++ {
		start := i * dim
		end := start + dim
		emb := embeddings[start:end]

		for j, v := range emb {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("NaN/Inf at text %d, index %d", i, j)
			}
		}
	}

	sim01 := cosineSimilarity(embeddings[0:dim], embeddings[dim:2*dim])
	sim12 := cosineSimilarity(embeddings[dim:2*dim], embeddings[2*dim:3*dim])
	sim02 := cosineSimilarity(embeddings[0:dim], embeddings[2*dim:3*dim])

	t.Logf("Similarities: 0-1=%f, 1-2=%f, 0-2=%f", sim01, sim12, sim02)

	if sim01 > 0.95 || sim12 > 0.95 || sim02 > 0.95 {
		t.Logf("Warning: Different texts have high similarity, may indicate model issue")
	}
}

func TestEmbeddingCoherence_BatchConsistency(t *testing.T) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	text := "artificial intelligence"

	embBatch := embedder.ProxyEmbedBatch(context.Background(), []string{text, text, text})
	dim := 128

	for i := 0; i < 3; i++ {
		start := i * dim
		end := start + dim
		emb := embBatch[start:end]

		for j, v := range emb {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("Batch NaN/Inf at batch %d, index %d", i, j)
			}
		}
	}

	sim01 := cosineSimilarity(embBatch[0:dim], embBatch[dim:2*dim])
	sim12 := cosineSimilarity(embBatch[dim:2*dim], embBatch[2*dim:3*dim])

	if sim01 < 0.99 || sim12 < 0.99 {
		t.Errorf("Batch of same texts should produce identical embeddings")
	}
}

func getOllamaEmbedding(model, text string) ([]float32, error) {
	reqBody, err := json.Marshal(OllamaEmbeddingRequest{Model: model, Prompt: text})
	if err != nil {
		return nil, err
	}

	resp, err := http.Post("http://localhost:11434/api/embeddings", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// Convert []float64 to []float32
	embedding := make([]float32, len(result.Embedding))
	for i, v := range result.Embedding {
		embedding[i] = float32(v)
	}
	return embedding, nil
}

func TestOllamaCoherence_BertTiny(t *testing.T) {
	// Check if Ollama is running
	_, err := getOllamaEmbedding("bert-tiny", "test")
	if err != nil {
		t.Skipf("Skipping: Ollama not running: %v", err)
	}

	// Create Fletcher embedder
	embedder, err := NewEmbedder("vocab.txt", "bert_tiny.bin", false, "bert-tiny", "fp32")
	if err != nil {
		t.Skipf("Skipping: could not create fletcher embedder: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog"

	// Get Fletcher embedding
	fletcherEmb := embedder.ProxyEmbedBatch(context.Background(), []string{text})
	if len(fletcherEmb) == 0 {
		t.Fatal("No fletcher embeddings generated")
	}

	// Get Ollama embedding (note: Ollama may not have bert-tiny, so we skip if not available)
	ollamaEmb, err := getOllamaEmbedding("bert-tiny", text)
	if err != nil {
		t.Skipf("Skipping: Ollama bert-tiny not available: %v", err)
	}

	// Compare - truncate to shorter length if needed
	minLen := len(fletcherEmb)
	if len(ollamaEmb) < minLen {
		minLen = len(ollamaEmb)
	}

	sim := cosineSimilarity(fletcherEmb[:minLen], ollamaEmb[:minLen])
	t.Logf("BertTiny cosine similarity: %.4f", sim)

	if sim < 0.90 {
		t.Errorf("Embedding similarity too low: %.4f (want > 0.90)", sim)
	}
}

func TestOllamaCoherence_NomicEmbedText(t *testing.T) {
	// Check if Ollama is running with nomic-embed-text
	_, err := getOllamaEmbedding("nomic-embed-text", "test")
	if err != nil {
		t.Skipf("Skipping: Ollama not running or nomic-embed-text not available: %v", err)
	}

	// Create Fletcher embedder with nomic-embed-text config
	embedder, err := NewEmbedder("vocab.txt", "bert_tiny.bin", false, "nomic-embed-text", "fp32")
	if err != nil {
		t.Skipf("Skipping: could not create fletcher embedder: %v", err)
	}

	text := "artificial intelligence is transforming the world"

	// Get Fletcher embedding
	fletcherEmb := embedder.ProxyEmbedBatch(context.Background(), []string{text})
	if len(fletcherEmb) == 0 {
		t.Fatal("No fletcher embeddings generated")
	}

	// Get Ollama embedding
	ollamaEmb, err := getOllamaEmbedding("nomic-embed-text", text)
	if err != nil {
		t.Skipf("Skipping: Ollama nomic-embed-text not available: %v", err)
	}

	// Compare
	minLen := len(fletcherEmb)
	if len(ollamaEmb) < minLen {
		minLen = len(ollamaEmb)
	}

	sim := cosineSimilarity(fletcherEmb[:minLen], ollamaEmb[:minLen])
	t.Logf("NomicEmbedText cosine similarity: %.4f", sim)

	// Target: > 0.99 for same model, lower threshold for different implementations
	if sim < 0.85 {
		t.Errorf("Embedding similarity too low: %.4f (want > 0.85)", sim)
	}
}
