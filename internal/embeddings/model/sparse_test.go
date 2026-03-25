package model

import (
	"testing"
)

func TestSparseEmbedding(t *testing.T) {
	emb := NewSparseEmbedding(30522)

	emb.Set(100, 1.5)
	emb.Set(200, 2.5)
	emb.Set(100, 3.0)

	if emb.Get(100) != 3.0 {
		t.Errorf("Expected 3.0, got %f", emb.Get(100))
	}

	if emb.NonZeroCount() != 2 {
		t.Errorf("Expected 2 non-zero, got %d", emb.NonZeroCount())
	}
}

func TestSparseEmbedding_ToDense(t *testing.T) {
	emb := NewSparseEmbedding(10)

	emb.Set(0, 1.0)
	emb.Set(5, 2.0)

	dense := emb.ToDense()

	if len(dense) != 10 {
		t.Errorf("Expected length 10, got %d", len(dense))
	}

	if dense[0] != 1.0 {
		t.Errorf("Expected dense[0]=1.0, got %f", dense[0])
	}

	if dense[5] != 2.0 {
		t.Errorf("Expected dense[5]=2.0, got %f", dense[5])
	}
}

func TestSparseConfig(t *testing.T) {
	config := DefaultSparseConfig()

	if config.VocabSize != 30522 {
		t.Errorf("Expected VocabSize 30522, got %d", config.VocabSize)
	}

	if config.HiddenSize != 128 {
		t.Errorf("Expected HiddenSize 128, got %d", config.HiddenSize)
	}
}

func TestSparseEmbedding_NegativeWeights(t *testing.T) {
	emb := NewSparseEmbedding(100)

	emb.Set(10, -1.0)
	emb.Set(20, 0.0)
	emb.Set(30, 1.0)

	if emb.NonZeroCount() != 1 {
		t.Errorf("Expected 1 non-zero (only positive weights), got %d", emb.NonZeroCount())
	}
}
