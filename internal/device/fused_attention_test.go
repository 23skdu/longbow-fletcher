//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// TestFusedAttentionCorrectness verifies fused attention produces same output as unfused
func TestFusedAttentionCorrectness(t *testing.T) {
	backend := NewMetalBackendFP16()
	
	testCases := []struct {
		name       string
		batch      int
		seqLen     int
		hiddenSize int
	}{
		{"Small_1x8x32", 1, 8, 32},
		{"Medium_2x16x64", 2, 16, 64},
		{"Large_4x32x128", 4, 32, 128},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			totalDim := tc.batch * tc.seqLen
			
			// Create test data
			qData := make([]float32, totalDim*tc.hiddenSize)
			kData := make([]float32, totalDim*tc.hiddenSize)
			vData := make([]float32, totalDim*tc.hiddenSize)
			
			// Initialize with deterministic values for reproducibility
			for i := range qData {
				qData[i] = float32(i%100) / 100.0
				kData[i] = float32((i+50)%100) / 100.0
				vData[i] = float32((i+25)%100) / 100.0
			}
			
			// Create tensors for unfused path
			q1 := backend.NewTensor(totalDim, tc.hiddenSize, qData)
			k1 := backend.NewTensor(totalDim, tc.hiddenSize, kData)
			v1 := backend.NewTensor(totalDim, tc.hiddenSize, vData)
			
			// Create tensors for fused path
			q2 := backend.NewTensor(totalDim, tc.hiddenSize, qData)
			k2 := backend.NewTensor(totalDim, tc.hiddenSize, kData)
			v2 := backend.NewTensor(totalDim, tc.hiddenSize, vData)
			
			scale := float32(1.0 / math.Sqrt(float64(tc.hiddenSize)))
			
			// Run unfused attention (current implementation)
			unfusedResult := q1.Attention(q1, k1, v1, tc.batch, tc.seqLen, 1, scale)
			backend.Synchronize()
			unfusedOutput := unfusedResult.ToHost()
			
			// Run fused attention (new implementation)
			// Note: This will call the fused kernel when implemented
			fusedResult := q2.(*MetalTensor).FusedAttention(q2, k2, v2, tc.batch, tc.seqLen, 1, scale)
			backend.Synchronize()
			fusedOutput := fusedResult.ToHost()
			
			// Compare outputs
			maxError := float32(0.0)
			for i := range unfusedOutput {
				diff := float32(math.Abs(float64(unfusedOutput[i] - fusedOutput[i])))
				if diff > maxError {
					maxError = diff
				}
			}
			
			// Allow small numerical differences due to FP16 precision
			tolerance := float32(1e-3)
			if maxError > tolerance {
				t.Errorf("Fused attention output differs from unfused: max error = %f (tolerance = %f)", 
					maxError, tolerance)
				
				// Print first few values for debugging
				t.Logf("First 10 unfused values: %v", unfusedOutput[:10])
				t.Logf("First 10 fused values:   %v", fusedOutput[:10])
			} else {
				t.Logf("Fused attention matches unfused (max error: %f)", maxError)
			}
			
			// Cleanup
			backend.PutTensor(unfusedResult)
			backend.PutTensor(fusedResult)
		})
	}
}

// BenchmarkFusedAttentionComparison compares fused vs unfused performance
func BenchmarkFusedAttentionComparison(b *testing.B) {
	backend := NewMetalBackendFP16()
	
	batch, seqLen, hiddenSize := 8, 128, 384
	totalDim := batch * seqLen
	
	// Create test data
	qData := make([]float32, totalDim*hiddenSize)
	kData := make([]float32, totalDim*hiddenSize)
	vData := make([]float32, totalDim*hiddenSize)
	for i := range qData {
		qData[i] = float32(i%100) / 100.0
		kData[i] = float32(i%100) / 100.0
		vData[i] = float32(i%100) / 100.0
	}
	
	scale := float32(1.0 / math.Sqrt(float64(hiddenSize)))
	
	b.Run("Unfused", func(b *testing.B) {
		q := backend.NewTensor(totalDim, hiddenSize, qData)
		k := backend.NewTensor(totalDim, hiddenSize, kData)
		v := backend.NewTensor(totalDim, hiddenSize, vData)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			result := q.Attention(q, k, v, batch, seqLen, 1, scale)
			backend.PutTensor(result)
		}
		backend.Synchronize()
	})
	
	b.Run("Fused", func(b *testing.B) {
		q := backend.NewTensor(totalDim, hiddenSize, qData)
		k := backend.NewTensor(totalDim, hiddenSize, kData)
		v := backend.NewTensor(totalDim, hiddenSize, vData)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			result := q.(*MetalTensor).FusedAttention(q, k, v, batch, seqLen, 1, scale)
			backend.PutTensor(result)
		}
		backend.Synchronize()
	})
}
