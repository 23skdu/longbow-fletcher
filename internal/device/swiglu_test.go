//go:build darwin && metal

package device

import (
	"math"
	"testing"

)

func TestSwiGLU_F16_Correctness(t *testing.T) {
	backend := NewMetalBackendFP16()
	
	n := 10
	interSize := 128 // Multiple of 4
	totalSize := n * interSize * 2
	
	data := make([]float32, totalSize)
	for i := range data {
		data[i] = float32(i%100) / 10.0
	}
	
	// input variable removed

	// We need to call the kernel directly or via LinearActivation if wired up.
	// LinearActivation uses it.
	
	// Create dummy weight/bias for LinearActivation to trigger SwiGLU
	// OR better, create a wrapper in *MetalTensor to call it directly for testing if possible.
	// But LinearActivation logic does: Linear -> Split -> Swish -> Mul.
	// Our Metal SwiGLU kernel handles the Split -> Swish -> Mul part given a combined input.
	// The implementation in metal_darwin.go for ActivationSwiGLU likely calls Metal_SwiGLU_F16.
	
	// Let's check `LinearActivation` logic in `metal_darwin.go`.
	// Ideally we just want to invoke `Metal_SwiGLU_F16`.
	// But `ActivationSwiGLU` is usually applied to the result of a Linear layer which is (N, 2*interSize).
	// So let's fake that.
	
	// We will use existing method: 
	// Make a tensor, call some method that triggers SwiGLU?
	// The device interface has `LinearActivation`.
	// But currently `MetalTensor` might not expose `SwiGLU` directly as a public method.
	// Let's assume we can use `LinearActivation` with Identity weights to isolate SwiGLU.
	
	// Setup Identity Map
	// Input: (N, 2*interSize)
	// Bias: 0
	// We want to pass this to LinearActivation... wait, LinearActivation computes Linear first.
	// Linear(x, w, b) -> (N, 2*interSize) -> SwiGLU -> (N, interSize).
	// If we want to test SwiGLU strictly, we need w to be Identity?
	// Too complex.
	
	// Better: use specialized test helper or checking `metal_darwin.go`
	// If `metal_darwin.go` has `func (t *MetalTensor) SwiGLU()`, we use that.
	// But likely it is handled inside `LinearActivation`.
	
	// Let's try to verify via `LinearActivation` with a simple "Pass-through" Linear.
	// If Input is (N, I), W is (I, 2*I).
	// If we set W s.t. Output is known, we can verify SwiGLU.
    // Or, simply verify that the logic runs without crash and produces deterministic output.
    
    // Actually, `metal_darwin.go` does NOT implement `SwiGLU` as a standalone `Activation` execution method usually.
    // Let's look at `metal_darwin.go` implementation of `LinearActivation`.
    // Assuming it calls `Metal_SwiGLU_F16`.
    
    // Verify against CPU reference.
    
    refOutput := make([]float32, n*interSize)
    for i := 0; i < n; i++ {
        for j := 0; j < interSize; j++ {
            idx := i*2*interSize + j
            x := data[idx]
            y := data[idx + interSize]
            swish := x / (1.0 + float32(math.Exp(float64(-x))))
            refOutput[i*interSize + j] = swish * y
        }
    }
    
    // We need to trigger the kernel.
    // Since we don't have a direct `t.SwiGLU()` method exposed in the interface or struct usually,
    // we might need to rely on `LinearActivation`.
    // Or we modify `metal_darwin.go` to add it if checking internal method.
    // But `LinearActivation` is the public path.
    
    // Construct simplified Linear: W=Identity (block), B=0.
    // Input X: (N, 2*interSize). 
    // Weight W: (2*interSize, 2*interSize).
    // This is getting large.
    
    // Alternative: Use `reflection` or rely on the underlying `callSwiGLU` if it exists?
    // Let's just create a small Linear that projects (N,1) -> (N, 2*interSize).
    // Input X (N,1) = 1.0.
    // Weight W (1, 2*interSize) = data.
    // Bias B = 0.
    // Result Linear = data.
    // Then SwiGLU is applied to data.
    
    inputVec := backend.NewTensor(n, 1, nil)
    inData := make([]float32, n)
    for i := range inData { inData[i] = 1.0 }
    inputVec.CopyFromFloat32(inData)
    
    // W must be (1, 2*interSize) to produce (N, 2*interSize) output.
    // We want the output of Linear to be `data`.
    // Since X=1, W=data row (broadcasted? No).
    // MatMul (N,1) x (1, 2*interSize) = (N, 2*interSize).
    // Row i of output = row i of X * W.
    // X[i] = 1.
    // So all rows of output will be identical to W.
    // That's not fully general test (batch variation lost).
    
    // How about (N, N) Identity x (N, 2*interSize).
    // X = Identity (N,N).
    // W = data (N, 2*interSize).
    // res = X * W = data.
    // Correct.
    
    X := backend.NewTensor(n, n, nil)
    xData := make([]float32, n*n)
    for i := 0; i < n; i++ { xData[i*n+i] = 1.0 }
    X.CopyFromFloat32(xData)
    
    W := backend.NewTensor(n, 2*interSize, data) // Use our data as weights
    
    // Bias zero
    B := backend.NewTensor(1, 2*interSize, nil) // Zeroes
    
    outTensor := X.LinearActivation(X, W, B, ActivationSwiGLU)
    outData := outTensor.ToHost()
    
    // Verify
    maxDiff := float32(0.0)
    for i := 0; i < len(outData); i++ {
        diff := float32(math.Abs(float64(outData[i] - refOutput[i])))
        if diff > maxDiff { maxDiff = diff }
    }
    
    t.Logf("Max SwiGLU Diff: %f", maxDiff)
    if maxDiff > 0.1 {
        t.Errorf("Mismatch > 0.1")
    }
}

func BenchmarkSwiGLU_F16(b *testing.B) {
	backend := NewMetalBackendFP16()
	
    // Simulate typical layer
	n := 32
	interSize := 4096 // Llama/Mistral size
    
    // Setup roughly
    X := backend.NewTensor(n, n, nil)
    W := backend.NewTensor(n, 2*interSize, nil)
    B := backend.NewTensor(1, 2*interSize, nil)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        out := X.LinearActivation(X, W, B, ActivationSwiGLU)
        backend.PutTensor(out)
    }
    backend.Synchronize()
}
