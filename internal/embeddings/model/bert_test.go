package model

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBertModelForward(t *testing.T) {
	config := BertConfig{
		VocabSize:             100,
		HiddenSize:            16,
		NumHiddenLayers:       1,
		NumAttentionHeads:     2,
		IntermediateSize:      32,
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

func TestBertConfigDefaults(t *testing.T) {
	cfg := DefaultBertTinyConfig()
	require.Equal(t, 30522, cfg.VocabSize)
	require.Equal(t, 128, cfg.HiddenSize)
	require.Equal(t, 2, cfg.NumHiddenLayers)

	cfg = DefaultNomicConfig()
	require.Equal(t, 30522, cfg.VocabSize)
	require.Equal(t, 768, cfg.HiddenSize)
	require.Equal(t, 12, cfg.NumHiddenLayers)
	require.Equal(t, PositionalRoPE, cfg.PositionEmbedding)
	require.Equal(t, true, cfg.FusedQKV)

	cfg = DefaultMiniLMConfig()
	require.Equal(t, 384, cfg.HiddenSize)
	require.Equal(t, 6, cfg.NumHiddenLayers)
	require.Equal(t, 12, cfg.NumAttentionHeads)
}

func TestModelConfigs(t *testing.T) {
	cfg := DefaultRoBERTaConfig()
	require.Equal(t, 50265, cfg.VocabSize)
	require.Equal(t, 768, cfg.HiddenSize)

	cfg = DefaultXLMRoBERTaConfig()
	require.Equal(t, 250002, cfg.VocabSize)
	require.Equal(t, 768, cfg.HiddenSize)

	cfg = DefaultBGEM3Config()
	require.Equal(t, 250002, cfg.VocabSize)
	require.Equal(t, 1024, cfg.HiddenSize)
	require.Equal(t, 24, cfg.NumHiddenLayers)

	cfg = DefaultE5MistralConfig()
	require.Equal(t, 32000, cfg.VocabSize)
	require.Equal(t, 4096, cfg.HiddenSize)
	require.Equal(t, 32, cfg.NumHiddenLayers)
	require.Equal(t, PositionalRoPE, cfg.PositionEmbedding)
}

func TestExtendedDimensionConfigs(t *testing.T) {
	cfg := DefaultDim1536Config()
	require.Equal(t, 1536, cfg.HiddenSize)
	require.Equal(t, 12, cfg.NumHiddenLayers)
	require.Equal(t, 12, cfg.NumAttentionHeads)
	require.Equal(t, 6144, cfg.IntermediateSize)

	cfg = DefaultDim2048Config()
	require.Equal(t, 2048, cfg.HiddenSize)
	require.Equal(t, 16, cfg.NumHiddenLayers)
	require.Equal(t, 16, cfg.NumAttentionHeads)
	require.Equal(t, 8192, cfg.IntermediateSize)

	cfg = DefaultDim3072Config()
	require.Equal(t, 3072, cfg.HiddenSize)
	require.Equal(t, 24, cfg.NumHiddenLayers)
	require.Equal(t, 24, cfg.NumAttentionHeads)
	require.Equal(t, 12288, cfg.IntermediateSize)
}

func TestBertModelForwardNonBatch(t *testing.T) {
	config := BertConfig{
		VocabSize:             100,
		HiddenSize:            16,
		NumHiddenLayers:       1,
		NumAttentionHeads:     2,
		IntermediateSize:      32,
		MaxPositionEmbeddings: 10,
	}

	model := NewBertModel(config)

	inputIDs := []int{1, 2, 3}
	lengths := []int{3}

	output := model.ForwardBatch(inputIDs, lengths)

	r, c := output.Dims()
	require.Equal(t, 1, r)
	require.Equal(t, config.HiddenSize, c)
}
