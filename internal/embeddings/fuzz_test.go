package embeddings

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/23skdu/longbow-fletcher/internal/device"
)

func isNaNOrInf(f float32) bool {
	return math.IsNaN(float64(f)) || math.IsInf(float64(f), 0)
}

func TestEmbedBatch_Fuzz_RandomASCII(t *testing.T) {
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
		"vocab.txt",
		"bert_tiny.bin",
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
		"vocab.txt",
		"bert_tiny.bin",
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
		"vocab.txt",
		"bert_tiny.bin",
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

	for j, v := range embeddings {
		if isNaNOrInf(v) {
			t.Errorf("Nomic model: NaN/Inf at index %d", j)
		}
	}
}

func TestFuzz_DimensionValidation(t *testing.T) {
	dimensions := []int{128, 384, 768, 1024, 1536, 2048, 3072}

	for _, dim := range dimensions {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			if !device.IsValidEmbeddingDimension(dim) {
				t.Errorf("Dimension %d should be valid", dim)
			}
		})
	}
}

func TestFuzz_InvalidDimensions(t *testing.T) {
	invalidDims := []int{0, -1, 1, 64, 256, 512, 1000, 1500, 4096, 8192}

	for _, dim := range invalidDims {
		if device.IsValidEmbeddingDimension(dim) {
			t.Errorf("Dimension %d should be invalid", dim)
		}
	}
}

func TestFuzz_DataTypeSizes(t *testing.T) {
	types := []device.DataType{
		device.Float32, device.Float16, device.Float64,
		device.Int8, device.Int16, device.Int32, device.Int64, device.Int,
		device.Uint8, device.Uint16, device.Uint32, device.Uint64, device.Uint, device.Uintptr,
		device.Complex64, device.Complex128,
	}

	for _, dt := range types {
		size := device.DataTypeSize(dt)
		if size <= 0 {
			t.Errorf("DataType %s has invalid size %d", dt.String(), size)
		}
	}
}

func TestFuzz_NumericOverflow(t *testing.T) {
	backend := device.NewCPUBackend()

	largeVal := float32(1e30)
	smallVal := float32(1e-30)

	t.Run("Add_LargeValues", func(t *testing.T) {
		a := backend.NewTensor(1, 4, []float32{largeVal, largeVal, largeVal, largeVal})
		b := backend.NewTensor(1, 4, []float32{largeVal, largeVal, largeVal, largeVal})
		a.Add(b)

		data := a.ToHost()
		for _, v := range data {
			if !isNaNOrInf(v) && math.Abs(float64(v)) < float64(largeVal) {
				t.Errorf("Expected large value, got %f", v)
			}
		}
	})

	t.Run("Scale_LargeValue", func(t *testing.T) {
		a := backend.NewTensor(1, 4, []float32{largeVal, largeVal, largeVal, largeVal})
		a.Scale(2.0)

		data := a.ToHost()
		for _, v := range data {
			if isNaNOrInf(v) {
				t.Logf("Got expected inf/nan for overflow: %f", v)
			}
		}
	})

	t.Run("Subnormal_SmallValue", func(t *testing.T) {
		a := backend.NewTensor(1, 4, []float32{smallVal, smallVal, smallVal, smallVal})
		b := backend.NewTensor(1, 4, []float32{smallVal, smallVal, smallVal, smallVal})
		a.Add(b)

		data := a.ToHost()
		for _, v := range data {
			if math.Abs(float64(v)) < float64(smallVal)/2 {
				t.Errorf("Expected value >= smallVal, got %f", v)
			}
		}
	})
}

func TestFuzz_AllDimensionsWork(t *testing.T) {
	dimensions := []int{128, 384, 768, 1024}

	for _, dim := range dimensions {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			backend := device.NewCPUBackend()
			tensor := backend.NewTensor(1, dim, nil)

			r, c := tensor.Dims()
			if r != 1 || c != dim {
				t.Errorf("Expected dims (1, %d), got (%d, %d)", dim, r, c)
			}

			for i := 0; i < dim; i++ {
				tensor.Set(0, i, float32(i))
			}

			data := tensor.ToHost()
			if len(data) != dim {
				t.Errorf("Expected %d elements, got %d", dim, len(data))
			}
		})
	}
}
