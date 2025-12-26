//go:build darwin && metal

package model

import (
	"testing"
	"github.com/23skdu/longbow-fletcher/internal/device"
)

func TestBertModel_Forward_Metal(t *testing.T) {
	// Setup Metal Backend
	backend := device.NewMetalBackend()
	if backend == nil {
		t.Fatal("Failed to create Metal backend")
	}

	config := BertConfig{
		VocabSize:     100,
		HiddenSize:    32, // Small size for speed
		NumHiddenLayers: 2,
		NumAttentionHeads: 4,
		IntermediateSize: 64,
		MaxPositionEmbeddings: 128,
	}

	model := NewBertModelWithBackend(config, backend)

	// Inputs
	inputIDs := []int{1, 5, 20, 9, 2} // Batch size 1, seq len 5
	lengths := []int{5}

	// Run Forward
	// This will use Metal for Embeddings, Encoder (Attention, FFN), Pooler
	output := model.ForwardBatch(inputIDs, lengths)
	
	// Check output dimensions
	r, c := output.Dims()
	if r != 1 {
		t.Errorf("Expected 1 row, got %d", r)
	}
	if c != config.HiddenSize {
		t.Errorf("Expected cols %d, got %d", config.HiddenSize, c)
	}
	
	// Check content (just no NaNs/Inf)
	data := output.ToHost()
	for i, v := range data {
		if v != v { // NaN check
			t.Errorf("NaN at index %d", i)
		}
	}
	
	t.Logf("Metal Forward Pass Successful. Output: %v", data[:5])
}
