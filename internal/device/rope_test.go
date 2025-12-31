//go:build darwin && metal

package device

import (
	"math"
	"math/rand"
	"testing"
)

func TestRoPE_F16_Correctness(t *testing.T) {
	backend := NewMetalBackendFP16()
	r := rand.New(rand.NewSource(1337))

	batchSize := 2
	seqLen := 128
	numHeads := 4
	headDim := 64 // Standard size
	
	totalSize := batchSize * seqLen * numHeads * headDim
	data := make([]float32, totalSize)
	for i := range data {
		data[i] = r.Float32()*2.0 - 1.0
	}
	
	tensor := backend.NewTensor(batchSize*seqLen, numHeads*headDim, data)
	
	// Apply RoPE on GPU
	tensor.ApplyRoPE(batchSize, seqLen, numHeads, headDim)
	gpuOut := tensor.ToHost()
	
	// Compute CPU Reference
	cpuOut := make([]float32, len(data))
	copy(cpuOut, data)
	
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			seqIdx := s // Position matches sequence index here
			for h := 0; h < numHeads; h++ {
				for i := 0; i < headDim/2; i++ {
					// Calculate theta
					theta := float64(seqIdx) * math.Pow(10000.0, -2.0*float64(i)/float64(headDim))
					cosTheta := float32(math.Cos(theta))
					sinTheta := float32(math.Sin(theta))
					
					offset := b*(seqLen*numHeads*headDim) + s*(numHeads*headDim) + h*headDim
					idx1 := offset + i
					idx2 := offset + i + headDim/2
					
					x1 := cpuOut[idx1]
					x2 := cpuOut[idx2]
					
					cpuOut[idx1] = x1*cosTheta - x2*sinTheta
					cpuOut[idx2] = x1*sinTheta + x2*cosTheta
				}
			}
		}
	}
	
	// Compare
	maxDiff := float32(0.0)
	for i := range gpuOut {
		diff := gpuOut[i] - cpuOut[i]
		if diff < 0 { diff = -diff }
		if diff > maxDiff { maxDiff = diff }
	}
	
	t.Logf("Max RoPE Diff: %f", maxDiff)
	
	// FP16 limits precision of trig functions. 
	// 5e-3 is usually good enough for embeddings.
	if maxDiff > 1e-2 {
		t.Errorf("RoPE mismatch > 1e-2")
	}
}

func BenchmarkRoPE_F16(b *testing.B) {
	backend := NewMetalBackendFP16()
	
	batchSize := 32
	seqLen := 512
	numHeads := 32
	headDim := 128
	
	tensor := backend.NewTensor(batchSize*seqLen, numHeads*headDim, nil)

	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.ApplyRoPE(batchSize, seqLen, numHeads, headDim)
	}
	backend.Synchronize()
}
