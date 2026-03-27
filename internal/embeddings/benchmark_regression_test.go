package embeddings

import (
	"context"
	"testing"

	"github.com/23skdu/longbow-fletcher/internal/device"
)

func BenchmarkRegression_Dimensions(b *testing.B) {
	dimensions := []int{128, 384, 768, 1024}

	for _, dim := range dimensions {
		b.Run("dim_"+string(rune(dim)), func(b *testing.B) {
			backend := device.NewCPUBackend()
			tensor := backend.NewTensor(1, dim, nil)

			for i := 0; i < dim; i++ {
				tensor.Set(0, i, float32(i))
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				data := tensor.ToHost()
				_ = data
			}
		})
	}
}

func BenchmarkRegression_BatchEmbedding(b *testing.B) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		b.Skipf("Skipping: could not create embedder: %v", err)
	}

	texts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Fletcher is a high performance embedding engine",
		"Machine learning models require careful optimization",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = embedder.ProxyEmbedBatch(context.Background(), texts)
	}
}

func BenchmarkRegression_Throughput(b *testing.B) {
	embedder, err := NewEmbedder(
		"vocab.txt",
		"bert_tiny.bin",
		false,
		"bert-tiny",
		"fp32",
	)
	if err != nil {
		b.Skipf("Skipping: could not create embedder: %v", err)
	}

	text := "The quick brown fox jumps over the lazy dog"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = embedder.ProxyEmbedBatch(context.Background(), []string{text})
	}
}

func TestRegression_DimensionConsistency(t *testing.T) {
	dimensions := []int{128, 384, 768, 1024}

	for _, dim := range dimensions {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			backend := device.NewCPUBackend()
			tensor := backend.NewTensor(1, dim, nil)

			for i := 0; i < dim; i++ {
				tensor.Set(0, i, float32(i))
			}

			data := tensor.ToHost()
			if len(data) != dim {
				t.Errorf("Dimension %d: expected %d elements, got %d", dim, dim, len(data))
			}

			for i := 0; i < dim; i++ {
				if data[i] != float32(i) {
					t.Errorf("Dimension %d: at index %d, expected %f, got %f", dim, i, float32(i), data[i])
				}
			}
		})
	}
}

func TestRegression_EmbeddingOutputStable(t *testing.T) {
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

	emb1 := embedder.ProxyEmbedBatch(context.Background(), []string{text})
	emb2 := embedder.ProxyEmbedBatch(context.Background(), []string{text})

	if len(emb1) != len(emb2) {
		t.Errorf("Embedding lengths differ: %d vs %d", len(emb1), len(emb2))
	}

	diff := 0.0
	for i := range emb1 {
		d := float64(emb1[i] - emb2[i])
		diff += d * d
	}

	if diff > 0.001 {
		t.Errorf("Embedding not stable: diff = %f (expected ~0)", diff)
	}
}
