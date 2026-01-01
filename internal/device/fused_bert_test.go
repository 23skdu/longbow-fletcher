//go:build metal

package device

import (
	"math"
	"testing"
)

func IsMetalAvailable() bool {
	return true
}

func TestFusedBertLayer(t *testing.T) {
	if !IsMetalAvailable() {
		t.Skip("Metal not available")
	}

	backend := NewMetalBackend()
	batchSize := 1
	seqLen := 4
	hiddenSize := 64 // Must be divisible by numHeads
	numHeads := 4
	headDim := hiddenSize / numHeads
	intermediateSize := 256
	eps := float32(1e-5)

	// Helper to create random tensor
	createRandom := func(r, c int) Tensor {
		d := make([]float32, r*c)
		for i := range d {
			d[i] = float32(i)*0.01 - 0.5 // deterministic pseudo-random
		}
		t32 := backend.NewTensor(r, c, d)
		return t32.Cast(Float16)
	}

	// --- Inputs ---
	input := createRandom(batchSize*seqLen, hiddenSize)
	
	// --- Weights ---
	wq := createRandom(hiddenSize, hiddenSize)
	wk := createRandom(hiddenSize, hiddenSize)
	wv := createRandom(hiddenSize, hiddenSize)
	wout := createRandom(hiddenSize, hiddenSize)
	winter := createRandom(hiddenSize, intermediateSize)
	woutffn := createRandom(intermediateSize, hiddenSize)

	// Biases
	bq := createRandom(1, hiddenSize)
	bk := createRandom(1, hiddenSize)
	bv := createRandom(1, hiddenSize)
	bout := createRandom(1, hiddenSize)
	binter := createRandom(1, intermediateSize)
	boutffn := createRandom(1, hiddenSize)

	// Norms
	gammaAttn := createRandom(1, hiddenSize)
	betaAttn := createRandom(1, hiddenSize)
	gammaFFN := createRandom(1, hiddenSize)
	betaFFN := createRandom(1, hiddenSize)

	// --- Run Fused Kernel ---
	// Cast Reference Inputs to F16 for Fused Call
	inputF16 := input.Cast(Float16)
	wqF16, wkF16, wvF16 := wq.Cast(Float16), wk.Cast(Float16), wv.Cast(Float16)
	woutF16, winterF16, woutffnF16 := wout.Cast(Float16), winter.Cast(Float16), woutffn.Cast(Float16)
	bqF16, bkF16, bvF16 := bq.Cast(Float16), bk.Cast(Float16), bv.Cast(Float16)
	boutF16, binterF16, boutffnF16 := bout.Cast(Float16), binter.Cast(Float16), boutffn.Cast(Float16)
	gammaAttnF16, betaAttnF16 := gammaAttn.Cast(Float16), betaAttn.Cast(Float16)
	gammaFFNF16, betaFFNF16 := gammaFFN.Cast(Float16), betaFFN.Cast(Float16)

	// Calculate fused (using F16 inputs)
	fusedResult := inputF16.FusedBertLayer(
		wqF16, wkF16, wvF16, woutF16, winterF16, woutffnF16,
		bqF16, bkF16, bvF16, boutF16, binterF16, boutffnF16,
		gammaAttnF16, betaAttnF16, gammaFFNF16, betaFFNF16,
		batchSize, seqLen, hiddenSize, numHeads, intermediateSize, eps,
	)
	
	// --- Run Sequential Reference (CPU) ---
	cpuBackend := NewCPUBackend()
	refInput := cpuBackend.NewTensor(batchSize*seqLen, hiddenSize, input.ToHost())
	refWq := cpuBackend.NewTensor(hiddenSize, hiddenSize, wq.ToHost())
	refWk := cpuBackend.NewTensor(hiddenSize, hiddenSize, wk.ToHost())
	refWv := cpuBackend.NewTensor(hiddenSize, hiddenSize, wv.ToHost())
	refWout := cpuBackend.NewTensor(hiddenSize, hiddenSize, wout.ToHost())
	refWinter := cpuBackend.NewTensor(hiddenSize, intermediateSize, winter.ToHost())
	refWoutFFN := cpuBackend.NewTensor(intermediateSize, hiddenSize, woutffn.ToHost())

	refBq := cpuBackend.NewTensor(1, hiddenSize, bq.ToHost())
	refBk := cpuBackend.NewTensor(1, hiddenSize, bk.ToHost())
	refBv := cpuBackend.NewTensor(1, hiddenSize, bv.ToHost())
	refBout := cpuBackend.NewTensor(1, hiddenSize, bout.ToHost())
	refBinter := cpuBackend.NewTensor(1, intermediateSize, binter.ToHost())
	refBoutFFN := cpuBackend.NewTensor(1, hiddenSize, boutffn.ToHost())
	
	refGammaAttn := cpuBackend.NewTensor(1, hiddenSize, gammaAttn.ToHost())
	refBetaAttn := cpuBackend.NewTensor(1, hiddenSize, betaAttn.ToHost())
	refGammaFFN := cpuBackend.NewTensor(1, hiddenSize, gammaFFN.ToHost())
	refBetaFFN := cpuBackend.NewTensor(1, hiddenSize, betaFFN.ToHost())

	// Run Ref on CPU
	qC := refInput.Linear(refInput, refWq, refBq)
	kC := refInput.Linear(refInput, refWk, refBk)
	vC := refInput.Linear(refInput, refWv, refBv)

	// Attn
	refAttnOut := refInput.Attention(qC, kC, vC, batchSize, seqLen, numHeads, 1.0/float32(math.Sqrt(float64(headDim))))

	// Out Proj
	projOutC := refInput.Linear(refAttnOut, refWout, refBout)

	// Residual 1
	res1C := refInput.Cast(Float32) // CPU uses F32
	res1C.Add(projOutC)

	// Norm 1
	norm1C := res1C.Cast(Float32)
	norm1C.LayerNorm(refGammaAttn, refBetaAttn, eps)

	// FFN
	interC := refInput.LinearActivation(norm1C, refWinter, refBinter, ActivationGELU)
	ffnOutC := refInput.Linear(interC, refWoutFFN, refBoutFFN)

	// Residual 2
	res2C := norm1C.Cast(Float32)
	res2C.Add(ffnOutC)

	// Norm 2
	finalRefC := res2C.Cast(Float32)
	finalRefC.LayerNorm(refGammaFFN, refBetaFFN, eps)
	
	refData := finalRefC.ToHost()
	fusedData := fusedResult.ToHost()
	// Verify
	
	debugIdx := 255
	if debugIdx >= len(refData) { debugIdx = len(refData) - 1 }
	if debugIdx < 0 { debugIdx = 0 }
	
	t.Logf("DEBUG: Ref[0]=%f, Ref[%d]=%f", refData[0], debugIdx, refData[debugIdx])
	t.Logf("DEBUG: Fused[0]=%f, Fused[%d]=%f", fusedData[0], debugIdx, fusedData[debugIdx])
	t.Logf("DEBUG: InputF16[0]=%f", input.ToHost()[0])
	
	if mt, ok := input.(*MetalTensor); ok {
		t.Logf("DEBUG: Go Input Address: 0x%x", mt.Address())
	}
	
	// --- Weights ---rify
	compareTensors(t, finalRefC, fusedResult, "fused", 0.15)
}

func compareTensors(t *testing.T, expected, actual Tensor, name string, tolerance float32) {
	expectedData := expected.ToHost()
	actualData := actual.ToHost()

	if len(expectedData) != len(actualData) {
		t.Fatalf("%s: Size mismatch: expected %d vs actual %d", name, len(expectedData), len(actualData))
	}

	var mismatches int
	maxDiff := float32(0.0)
	for i := range actualData {
		diff := float32(math.Abs(float64(actualData[i] - expectedData[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tolerance {
			t.Errorf("Mismatch at %d: fused %f vs ref %f (diff %f)", i, actualData[i], expectedData[i], diff)
			mismatches++
			if mismatches > 10 { break }
		}
	}
	t.Logf("Max Diff: %f", maxDiff)
	
	if maxDiff > tolerance {
		t.Fail()
	}
}
