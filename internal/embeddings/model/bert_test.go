package model

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBertModelForward(t *testing.T) {
	config := BertConfig{
		VocabSize:            100,
		HiddenSize:           16,
		NumHiddenLayers:      1,
		NumAttentionHeads:    2,
		IntermediateSize:     32,
		MaxPositionEmbeddings: 10,
	}
	
	model := NewBertModel(config)
	
	// Initialize with some non-zero values for testing
	// Simple random-like initialization
	for i := 0; i < config.VocabSize; i++ {
		for j := 0; j < config.HiddenSize; j++ {
			model.Embeddings.WordEmbeddings.Set(i, j, float32(i+j+1)/100.0)
		}
	}
	for i := 0; i < config.HiddenSize; i++ {
		for j := 0; j < config.HiddenSize; j++ {
			model.Pooler.Dense.Set(i, j, 1.0)
		}
		model.Pooler.Bias.Set(0, i, 0.1)
	}

	inputIDs := []int{1, 2, 3}
	output := model.Forward(inputIDs)
	
	r, c := output.Dims()
	require.Equal(t, 1, r, "Pooler should return 1xH output")
	require.Equal(t, config.HiddenSize, c)
	
	// Check if output is not all zeros
	data := output.ToHost()
	hasNonZero := false
	for _, v := range data {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	require.True(t, hasNonZero, "Output should not be all zeros")
}
