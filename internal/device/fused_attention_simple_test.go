//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestFusedAttentionBasic tests just the fused path without comparison
func TestFusedAttentionBasic(t *testing.T) {
	backend := NewMetalBackendFP16()
	
	batch, seqLen, hiddenSize := 1, 8, 32
	totalDim := batch * seqLen
	
	// Create test data
	qData := make([]float32, totalDim*hiddenSize)
	kData := make([]float32, totalDim*hiddenSize)
	vData := make([]float32, totalDim*hiddenSize)
	
	for i := range qData {
		qData[i] = 0.1
		kData[i] = 0.1
		vData[i] = 0.1
	}
	
	q := backend.NewTensor(totalDim, hiddenSize, qData)
	k := backend.NewTensor(totalDim, hiddenSize, kData)
	v := backend.NewTensor(totalDim, hiddenSize, vData)
	
	scale := float32(1.0 / math.Sqrt(float64(hiddenSize)))
	
	// Run fused attention
	result := q.(*MetalTensor).FusedAttention(q, k, v, batch, seqLen, scale)
	backend.Synchronize()
	output := result.ToHost()
	
	t.Logf("Output shape: %d elements", len(output))
	t.Logf("First 10 values: %v", output[:10])
	
	// Basic sanity check - output should not be all zeros
	nonZero := 0
	for _, v := range output {
		if v != 0 {
			nonZero++
		}
	}
	
	if nonZero == 0 {
		t.Error("Fused attention output is all zeros")
	} else {
		t.Logf("Non-zero elements: %d/%d", nonZero, len(output))
	}
	
	backend.PutTensor(result)
}
