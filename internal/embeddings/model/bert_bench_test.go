package model

import (
	"testing"
	"github.com/23skdu/longbow-fletcher/internal/device"
)

func BenchmarkBertModel_Forward_CPU(b *testing.B) {
	// Config matching prajjwal1/bert-tiny (L=2, H=128)
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

	// Input setup: 64 token sequence
	seqLen := 64
	inputIDs := make([]int, seqLen)
	for i := range inputIDs {
		inputIDs[i] = i % 1000
	}
	lengths := []int{seqLen}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := model.ForwardBatch(inputIDs, lengths)
		backend.PutTensor(out)
	}
}

func BenchmarkBertModel_Forward_Metal(b *testing.B) {
	defer func() {
		if r := recover(); r != nil {
			b.Skip("Metal backend not available")
		}
	}()

	config := BertConfig{
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		VocabSize:             30522,
		MaxPositionEmbeddings: 512,
	}

	backend := device.NewMetalBackendFP16()
	model := NewBertModelWithBackend(config, backend)

	// Input setup: 64 token sequence
	seqLen := 64
	inputIDs := make([]int, seqLen)
	for i := range inputIDs {
		inputIDs[i] = i % 1000
	}
	lengths := []int{seqLen}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.ForwardBatch(inputIDs, lengths)
		// Synchronize to measure actual computation time + overhead
		backend.Synchronize()
	}
}

func BenchmarkBertModel_Forward_CPU_Batch32(b *testing.B) {
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

	// Input setup: Batch 32, SeqLen 64
	batchSize := 32
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

func BenchmarkBertModel_Forward_Metal_Batch32(b *testing.B) {
	defer func() {
		if r := recover(); r != nil {
			b.Skip("Metal backend not available")
		}
	}()

	config := BertConfig{
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		VocabSize:             30522,
		MaxPositionEmbeddings: 512,
	}

	backend := device.NewMetalBackendFP16()
	model := NewBertModelWithBackend(config, backend)

	// Input setup: Batch 32, SeqLen 64
	batchSize := 32
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
		model.ForwardBatch(inputIDs, lengths)
		backend.Synchronize()
	}
}
