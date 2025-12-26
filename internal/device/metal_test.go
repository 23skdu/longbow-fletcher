//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestMetalBackend_Add(t *testing.T) {
	b := NewMetalBackend()
	t1 := b.NewTensor(2, 2, []float32{1, 2, 3, 4})
	t2 := b.NewTensor(2, 2, []float32{10, 20, 30, 40})
	
	t1.Add(t2)
	
	out := t1.ToHost()
	expected := []float32{11, 22, 33, 44}
	
	for i, v := range out {
		if math.Abs(float64(v - expected[i])) > 1e-5 {
			t.Errorf("Add mismatch at %d: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestMetalBackend_Mul(t *testing.T) {
	b := NewMetalBackend()
	// A: 2x3
	// B: 3x2
	// C: 2x2
	
	aData := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	bData := []float32{
		7, 8,
		9, 10,
		11, 12,
	}
	
	A := b.NewTensor(2, 3, aData)
	B := b.NewTensor(3, 2, bData)
	C := b.NewTensor(2, 2, nil)
	
	C.Mul(A, B)
	
	// Expected:
	// row 0: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	// row 0: 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
	// row 1: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	// row 1: 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
	
	out := C.ToHost()
	expected := []float32{58, 64, 139, 154}
	
	for i, v := range out {
		if math.Abs(float64(v - expected[i])) > 1e-4 {
			t.Errorf("Mul mismatch at %d: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestMetalBackend_Softmax(t *testing.T) {
	b := NewMetalBackend()
	// 1x3
	data := []float32{1.0, 2.0, 3.0}
	tensor := b.NewTensor(1, 3, data)
	
	tensor.Softmax()
	
	out := tensor.ToHost()
	
	// exp(1)=2.718, exp(2)=7.389, exp(3)=20.085. Sum=30.192
	// p1 = 0.09
	// p2 = 0.2447
	// p3 = 0.6652
	
	sum := 0.0
	for _, v := range out {
		sum += float64(v)
	}
	
	if math.Abs(sum - 1.0) > 1e-4 {
		t.Errorf("Softmax sum not 1.0: %f", sum)
	}
	
	if out[2] < out[1] || out[1] < out[0] {
		t.Errorf("Softmax order mismatch: %v", out)
	}
}

func TestMetalBackend_LayerNorm(t *testing.T) {
	b := NewMetalBackend()
	rows, cols := 2, 4
	data := []float32{
		1, 2, 3, 4,
		10, 20, 30, 40,
	}
	// Mean 1: 2.5. Var 1: 1.25. Std: ~1.118
	// Mean 2: 25. Var 2: 125. Std: ~11.18
	
	t1 := b.NewTensor(rows, cols, data)
	gamma := b.NewTensor(1, cols, []float32{1, 1, 1, 1})
	beta := b.NewTensor(1, cols, []float32{0, 0, 0, 0})
	
	t1.LayerNorm(gamma, beta, 1e-5)
	
	out := t1.ToHost()
	
	// Check first row mean approx 0, std approx 1
	m1 := 0.0
	for i := 0; i < 4; i++ { m1 += float64(out[i]) }
	m1 /= 4
	
	if math.Abs(m1) > 1e-3 {
		t.Errorf("LayerNorm mean not 0: %f", m1)
	}
}

func TestMetalBackend_AddBias(t *testing.T) {
	b := NewMetalBackend()
	rows, cols := 2, 3
	data := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	bias := []float32{10, 20, 30}
	biasTensor := b.NewTensor(1, cols, bias)
	
	tensor := b.NewTensor(rows, cols, data)
	tensor.AddBias(biasTensor)
	
	out := tensor.ToHost()
	expected := []float32{
		11, 22, 33,
		14, 25, 36,
	}
	
	for i, v := range out {
		if math.Abs(float64(v - expected[i])) > 1e-5 {
			t.Errorf("AddBias mismatch at %d: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestMetalBackend_SliceInternal(t *testing.T) {
	// Test the Copy-based Slice implementation
	b := NewMetalBackend()
	rows, cols := 4, 2
	data := []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}
	tensor := b.NewTensor(rows, cols, data)
	
	// Slice rows 1 to 3 (exclusive) -> rows 1, 2
	slice := tensor.Slice(1, 3, 0, 2)
	
	r, c := slice.Dims()
	if r != 2 || c != 2 {
		t.Fatalf("Slice dims wrong: %dx%d", r, c)
	}
	
	out := slice.ToHost()
	expected := []float32{
		3, 4,
		5, 6,
	}
	
	for i, v := range out {
		if math.Abs(float64(v - expected[i])) > 1e-5 {
			t.Errorf("Slice mismatch at %d: got %f, want %f", i, v, expected[i])
		}
	}
	
	// Verify it's a view? (writes affect original?)
	// Not easily testable if ToHost implicitly syncs. 
	// But AddScalar to slice should affect original.
	slice.AddScalar(100.0)
	
	outOriginal := tensor.ToHost()
	// Row 1, 2 should have +100
	// Row 0, 3 should be unchanged
	if outOriginal[2] < 100 { // Index 2 is start of row 1 (3+100=103)
		t.Errorf("Slice write-back failed. Value: %f", outOriginal[2]) 
	}
}

func TestMetalBackend_Gather(t *testing.T) {
	b := NewMetalBackend()
	rows, cols := 5, 2
	// Table: 
	// 0: [0, 1]
	// 1: [2, 3]
	// 2: [4, 5]
	// 3: [6, 7]
	// 4: [8, 9]
	data := make([]float32, rows*cols)
	for i := range data {
		data[i] = float32(i)
	}
	table := b.NewTensor(rows, cols, data)
	
	indices := []int{1, 4, 0}
	// Expected:
	// [2, 3]
	// [8, 9]
	// [0, 1]
	
	gathered := table.(*MetalTensor).Gather(indices)
	
	r, c := gathered.Dims()
	if r != 3 || c != 2 {
		t.Fatalf("Gather dims wrong: %dx%d", r, c)
	}
	
	out := gathered.ToHost()
	expected := []float32{
		2, 3,
		8, 9,
		0, 1,
	}
	
	for i, v := range out {
		if math.Abs(float64(v - expected[i])) > 1e-5 {
			t.Errorf("Gather mismatch at %d: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestMetalBackend_Set(t *testing.T) {
	b := NewMetalBackend()
	rows, cols := 2, 2
	tensor := b.NewTensor(rows, cols, nil) // Zeros
	
	// Set 0,0 to 1.0
	tensor.Set(0, 0, 1.0)
	// Set 1,1 to 2.0
	tensor.Set(1, 1, 2.0)
	
	out := tensor.ToHost()
	// Expected [1, 0, 0, 2]
	
	if out[0] != 1.0 {
		t.Errorf("Set failed at 0,0. Got %f", out[0])
	}
	if out[3] != 2.0 {
		t.Errorf("Set failed at 1,1. Got %f", out[3])
	}
}
