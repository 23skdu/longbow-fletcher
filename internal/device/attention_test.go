//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

// simpleAttentionCPU is a reference implementation of multi-head Attention.
// Scores = Softmax(Q * K^T * scale)
// Result = Scores * V
func simpleAttentionCPU(q, k, v []float32, batch, seq, headDim, numHeads int, scale float32) []float32 {
	hiddenSize := headDim * numHeads
	result := make([]float32, batch*seq*hiddenSize)
	
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			scores := make([]float32, seq*seq)
			// Q * K^T for this head
			for i := 0; i < seq; i++ {
				for j := 0; j < seq; j++ {
					sum := float32(0.0)
					for d := 0; d < headDim; d++ {
						qVal := q[b*seq*hiddenSize + i*hiddenSize + h*headDim + d]
						kVal := k[b*seq*hiddenSize + j*hiddenSize + h*headDim + d]
						sum += qVal * kVal
					}
					scores[i*seq+j] = sum * scale
				}
			}
			
			// Softmax for this head
			for i := 0; i < seq; i++ {
				maxVal := float32(-math.MaxFloat32)
				for j := 0; j < seq; j++ {
					if scores[i*seq+j] > maxVal {
						maxVal = scores[i*seq+j]
					}
				}
				sumExp := float32(0.0)
				for j := 0; j < seq; j++ {
					val := float32(math.Exp(float64(scores[i*seq+j] - maxVal)))
					scores[i*seq+j] = val
					sumExp += val
				}
				for j := 0; j < seq; j++ {
					scores[i*seq+j] /= sumExp
				}
			}
			
			// Scores * V for this head
			for i := 0; i < seq; i++ {
				for d := 0; d < headDim; d++ {
					sum := float32(0.0)
					for j := 0; j < seq; j++ {
						score := scores[i*seq+j]
						vVal := v[b*seq*hiddenSize + j*hiddenSize + h*headDim + d]
						sum += score * vVal
					}
					result[b*seq*hiddenSize + i*hiddenSize + h*headDim + d] = sum
				}
			}
		}
	}
	
	return result
}

func TestMetalAttention_Precision(t *testing.T) {
	b := NewMetalBackendFP16()
	
	batch := 1
	seq := 4
	numHeads := 2
	headDim := 4
	hiddenSize := numHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	
	qData := make([]float32, batch*seq*hiddenSize)
	kData := make([]float32, batch*seq*hiddenSize)
	vData := make([]float32, batch*seq*hiddenSize)
	
	// Initialize with deterministic values
	for i := range qData {
		qData[i] = float32(i) * 0.1
		kData[i] = float32(len(kData)-i) * 0.1
		vData[i] = float32(i%3) * 0.5
	}
	
	tmQ := b.NewTensor(batch*seq, hiddenSize, qData)
	tmK := b.NewTensor(batch*seq, hiddenSize, kData)
	tmV := b.NewTensor(batch*seq, hiddenSize, vData)
	
	// The Attention method signature:
	// func (t *MetalTensor) Attention(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor
	res := tmQ.(*MetalTensor).Attention(tmQ, tmK, tmV, batch, seq, numHeads, scale)
	b.Synchronize()
	
	got := res.Data()
	want := simpleAttentionCPU(qData, kData, vData, batch, seq, headDim, numHeads, scale)
	
	t.Logf("Got (first 8): %v", got[:8])
	t.Logf("Want (first 8): %v", want[:8])

	
	// Compare
	mse := 0.0
	maxDiff := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		mse += diff * diff
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	mse /= float64(len(got))
	
	t.Logf("MSE: %v, MaxDiff: %v", mse, maxDiff)
	
	// FP16 tolerance is looser, 1e-3 is reasonable for small inputs
	if mse > 1e-3 {
		t.Errorf("MSE too high: %v > 1e-3", mse)
	}
}
