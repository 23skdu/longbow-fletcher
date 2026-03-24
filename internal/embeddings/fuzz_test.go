package embeddings

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

func isNaNOrInf(f float32) bool {
	return math.IsNaN(float64(f)) || math.IsInf(float64(f), 0)
}

func TestEmbedBatch_Fuzz_RandomASCII(t *testing.T) {
	embedder, err := NewEmbedder(
		"tokenizer.json",
		"bert_tiny.safetensors",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	for i := 0; i < 100; i++ {
		length := rand.Intn(500) + 1
		text := generateRandomASCII(length)

		embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{text})

		for j, v := range embeddings {
			if isNaNOrInf(v) {
				t.Errorf("Fuzz test failed: NaN/Inf at index %d for text length %d", j, length)
			}
		}

		expectedDim := 128
		if len(embeddings) != expectedDim {
			t.Errorf("Embedding dimension mismatch: expected %d, got %d", expectedDim, len(embeddings))
		}
	}
}

func TestEmbedBatch_Fuzz_RandomUTF8(t *testing.T) {
	embedder, err := NewEmbedder(
		"tokenizer.json",
		"bert_tiny.safetensors",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	for i := 0; i < 100; i++ {
		length := rand.Intn(200) + 1
		text := generateRandomUTF8(length)

		embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{text})

		for j, v := range embeddings {
			if isNaNOrInf(v) {
				t.Errorf("Fuzz test failed (UTF8): NaN/Inf at index %d for text length %d", j, length)
			}
		}
	}
}

func TestEmbedBatch_Fuzz_EmptyAndEdge(t *testing.T) {
	embedder, err := NewEmbedder(
		"tokenizer.json",
		"bert_tiny.safetensors",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	testCases := []string{
		"",
		"a",
		" ",
		"\n",
		"\t",
		"   ",
		"0123456789",
		"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
	}

	for _, text := range testCases {
		embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{text})

		if len(embeddings) == 0 {
			t.Errorf("Empty/edge case returned no embeddings for input: %q", text)
		}

		for j, v := range embeddings {
			if isNaNOrInf(v) {
				t.Errorf("NaN/Inf at index %d for edge case %q", j, text)
			}
		}
	}
}

func TestEmbedBatch_Fuzz_LongSequence(t *testing.T) {
	embedder, err := NewEmbedder(
		"tokenizer.json",
		"bert_tiny.safetensors",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	longText := ""
	for i := 0; i < 1000; i++ {
		longText += "The quick brown fox jumps over the lazy dog. "
	}

	embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{longText})

	for j, v := range embeddings {
		if isNaNOrInf(v) {
			t.Errorf("Long sequence: NaN/Inf at index %d", j)
		}
	}
}

func generateRandomASCII(length int) string {
	const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-"
	result := make([]byte, length)
	for i := 0; i < length; i++ {
		result[i] = chars[rand.Intn(len(chars))]
	}
	return string(result)
}

func generateRandomUTF8(length int) string {
	result := make([]rune, length)
	for i := 0; i < length; i++ {
		result[i] = rune(rand.Intn(0x10FFFF))
	}
	return string(result)
}

func TestEmbeddings_BatchVariedLengths(t *testing.T) {
	embedder, err := NewEmbedder(
		"tokenizer.json",
		"bert_tiny.safetensors",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	texts := []string{
		"a",
		"short text",
		"this is a medium length text for testing",
		"this is a much longer text that contains many more words and should produce a larger embedding vector when processed by the transformer model",
	}

	embeddings := embedder.ProxyEmbedBatch(context.Background(), texts)

	dim := 128
	expectedLen := len(texts) * dim

	if len(embeddings) != expectedLen {
		t.Errorf("Batch embedding length mismatch: expected %d, got %d", expectedLen, len(embeddings))
	}

	for j, v := range embeddings {
		if isNaNOrInf(v) {
			t.Errorf("Batch test: NaN/Inf at index %d", j)
		}
	}
}

func TestEmbeddings_NomicModel(t *testing.T) {
	embedder, err := NewEmbedder(
		"tokenizer.json",
		"bert_tiny.safetensors",
		false,
		"nomic-embed-text",
		"fp32",
	)
	if err != nil {
		t.Skipf("Skipping: could not create embedder: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog"
	embeddings := embedder.ProxyEmbedBatch(context.Background(), []string{text})

	for j, v := range embeddings {
		if isNaNOrInf(v) {
			t.Errorf("Nomic model: NaN/Inf at index %d", j)
		}
	}
}
