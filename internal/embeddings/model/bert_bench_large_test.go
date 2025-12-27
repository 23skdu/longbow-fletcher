package model

import (
	"testing"
	"github.com/23skdu/longbow-fletcher/internal/device"
)

func BenchmarkBertModel_Forward_CPU_Batch64(b *testing.B) {
	config := BertConfig{
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		VocabSize:             30522,
		MaxPositionEmbeddings: 512,
	}

	backend := device.NewCPUBackend()
	model := NewBertModelWithBackend(config, backend)

	batchSize := 64
	seqLen := 64
	totalLen := batchSize * seqLen
	inputIDs := make([]int, totalLen)
	lengths := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		lengths[i] = seqLen
		for j := 0; j < seqLen; j++ {
			inputIDs[i*seqLen+j] = (i*seqLen + j) % 1000
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := model.ForwardBatch(inputIDs, lengths)
		backend.PutTensor(out)
	}
}

func BenchmarkBertModel_Forward_CPU_Batch128(b *testing.B) {
	config := BertConfig{
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		VocabSize:             30522,
		MaxPositionEmbeddings: 512,
	}

	backend := device.NewCPUBackend()
	model := NewBertModelWithBackend(config, backend)

	batchSize := 128
	seqLen := 64
	totalLen := batchSize * seqLen
	inputIDs := make([]int, totalLen)
	lengths := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		lengths[i] = seqLen
		for j := 0; j < seqLen; j++ {
			inputIDs[i*seqLen+j] = (i*seqLen + j) % 1000
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := model.ForwardBatch(inputIDs, lengths)
		backend.PutTensor(out)
	}
}
