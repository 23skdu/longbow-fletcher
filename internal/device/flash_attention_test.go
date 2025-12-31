//go:build darwin && metal

package device

import (
	"math/rand"
	"testing"
)

func TestFlashAttention_Consistency(t *testing.T) {
	// Setup Backend FP16
	// We need to ensure we run on a machine with Metal support, but the build tag handles compilation.
	// Runtime check might be needed if running on non-Apple Silicon Mac, but usually assumed if build tag matches.
	
	// Create Backend
	b := NewMetalBackendFP16()
	
	batchSize := 2
	seqLen := 128 // Multiple of 32 to utilize blocks fully
	numHeads := 4
	headDim := 64
	hiddenSize := numHeads * headDim
	scale := float32(0.125) // 1/sqrt(64)
	
	// Create random data
	size := batchSize * seqLen * hiddenSize
	qData := make([]float32, size)
	kData := make([]float32, size)
	vData := make([]float32, size)
	
	for i := range qData {
		qData[i] = rand.Float32()*2.0 - 1.0 // -1 to 1
		kData[i] = rand.Float32()*2.0 - 1.0
		vData[i] = rand.Float32()*2.0 - 1.0
	}
	
	q := b.NewTensor(batchSize*seqLen, hiddenSize, qData)
	k := b.NewTensor(batchSize*seqLen, hiddenSize, kData)
	v := b.NewTensor(batchSize*seqLen, hiddenSize, vData)
	
	mq := q.(*MetalTensor)
	
	// Run Reference (Graph)
	// AttentionGraph(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor
	ref := mq.AttentionGraph(q, k, v, batchSize, seqLen, numHeads, scale)
	refData := ref.ToHost()
	
	// Run Flash
	flash := mq.FlashAttention(q, k, v, batchSize, seqLen, numHeads, scale)
	flashData := flash.ToHost()
	
	// Compare
	if len(refData) != len(flashData) {
		t.Fatalf("Size mismatch: %d vs %d", len(refData), len(flashData))
	}
	
	maxDiff := float32(0.0)
	meanDiff := float32(0.0)
	maxIdx := 0
	
	for i := range refData {
		diff := refData[i] - flashData[i]
		if diff < 0 { diff = -diff }
		if diff > maxDiff { 
			maxDiff = diff 
			maxIdx = i
		}
		meanDiff += diff
	}
	meanDiff /= float32(len(refData))
	
	t.Logf("Max Diff: %f at index %d (Ref=%f, Flash=%f)", maxDiff, maxIdx, refData[maxIdx], flashData[maxIdx])
	t.Logf("Mean Diff: %f", meanDiff)
	
	// Check relative error if values are large
	refVal := refData[maxIdx]
	if refVal < 0 { refVal = -refVal }
	relError := maxDiff / (refVal + 1e-6)
	t.Logf("Relative Error at Max Diff: %f", relError)
	
	// Tolerance: 0.25 (roughly 5-10% depends on magnitude)
	// Given random inputs and FP16, divergence is possible.
	if maxDiff > 0.25 { 
		t.Errorf("Max diff %f exceeds tolerance 0.25", maxDiff)
	}
}
