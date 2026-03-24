//go:build linux && cuda

package device

import (
	"math"
	"testing"
)

func TestCudaTensor_Add(t *testing.T) {
	backend := NewCudaBackend()

	a := backend.NewTensor(2, 3, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	b := backend.NewTensor(2, 3, []float32{
		10, 20, 30,
		40, 50, 60,
	})

	a.Add(b)

	result := a.ToHost()

	expected := []float32{
		11, 22, 33,
		44, 55, 66,
	}

	for i, e := range expected {
		if math.Abs(float64(result[i]-e)) > 0.001 {
			t.Errorf("Add: expected %v, got %v", expected, result)
			break
		}
	}
}

func TestCudaTensor_AddScalar(t *testing.T) {
	backend := NewCudaBackend()

	a := backend.NewTensor(2, 2, []float32{
		1, 2,
		3, 4,
	})

	a.AddScalar(10)

	result := a.ToHost()

	expected := []float32{
		11, 12,
		13, 14,
	}

	for i, e := range expected {
		if math.Abs(float64(result[i]-e)) > 0.001 {
			t.Errorf("AddScalar: expected %v, got %v", expected, result)
			break
		}
	}
}

func TestCudaTensor_Scale(t *testing.T) {
	backend := NewCudaBackend()

	a := backend.NewTensor(2, 2, []float32{
		1, 2,
		3, 4,
	})

	a.Scale(2)

	result := a.ToHost()

	expected := []float32{
		2, 4,
		6, 8,
	}

	for i, e := range expected {
		if math.Abs(float64(result[i]-e)) > 0.001 {
			t.Errorf("Scale: expected %v, got %v", expected, result)
			break
		}
	}
}

func TestCudaTensor_Tanh(t *testing.T) {
	backend := NewCudaBackend()

	a := backend.NewTensor(1, 3, []float32{
		0, 1, -1,
	})

	a.Tanh()

	result := a.ToHost()

	// tanh(0) = 0, tanh(1) ≈ 0.761594, tanh(-1) ≈ -0.761594
	if math.Abs(float64(result[0])) > 0.001 {
		t.Errorf("Tanh(0): expected ~0, got %v", result[0])
	}
	if math.Abs(float64(result[1]-0.761594)) > 0.01 {
		t.Errorf("Tanh(1): expected ~0.76, got %v", result[1])
	}
	if math.Abs(float64(result[2]+0.761594)) > 0.01 {
		t.Errorf("Tanh(-1): expected ~-0.76, got %v", result[2])
	}
}

func TestCudaTensor_Slice(t *testing.T) {
	backend := NewCudaBackend()

	// Create 4x4 matrix
	data := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	a := backend.NewTensor(4, 4, data)

	// Slice rows 1-3, cols 1-3 (2x2 submatrix starting at [1,1])
	// Should extract: 6,7; 10,11
	sliced := a.Slice(1, 3, 1, 3)

	r, c := sliced.Dims()
	if r != 2 || c != 2 {
		t.Errorf("Slice dims: expected 2x2, got %dx%d", r, c)
	}

	result := sliced.ToHost()
	expected := []float32{
		6, 7,
		10, 11,
	}

	for i, e := range expected {
		if math.Abs(float64(result[i]-e)) > 0.001 {
			t.Errorf("Slice: expected %v, got %v", expected, result)
			break
		}
	}
}

func TestCudaTensor_Cast_FP16_to_FP32(t *testing.T) {
	backend := NewCudaBackendFP16()

	// Create FP16 tensor
	data := []float32{
		1.0, -2.0, 0.5,
	}
	a := backend.NewTensor(1, 3, data)

	// Cast to FP32
	b := a.Cast(Float32)

	result := b.ToHost()

	for i, e := range data {
		if math.Abs(float64(result[i]-e)) > 0.01 {
			t.Errorf("Cast FP16->FP32: expected %v, got %v", e, result[i])
		}
	}
}

func TestCudaTensor_Cast_FP32_to_FP16(t *testing.T) {
	backend := NewCudaBackend()

	// Create FP32 tensor
	data := []float32{
		1.0, -2.0, 0.5,
	}
	a := backend.NewTensor(1, 3, data)

	// Cast to FP16
	b := a.Cast(Float16)

	result := b.ToHost()

	for i, e := range data {
		if math.Abs(float64(result[i]-e)) > 0.01 {
			t.Errorf("Cast FP32->FP16: expected %v, got %v", e, result[i])
		}
	}
}

func TestCudaBackend_GetVRAMUsage(t *testing.T) {
	backend := NewCudaBackend()

	allocated, total := backend.GetVRAMUsage()

	if total <= 0 {
		t.Errorf("GetVRAMUsage: expected total > 0, got %d", total)
	}

	if allocated < 0 {
		t.Errorf("GetVRAMUsage: expected allocated >= 0, got %d", allocated)
	}

	t.Logf("VRAM: allocated=%d, total=%d", allocated, total)
}

func TestCudaTensor_Paste(t *testing.T) {
	backend := NewCudaBackend()

	// Create destination 3x3 matrix (zeros)
	dst := backend.NewTensor(3, 3, nil)

	// Create source 2x2 matrix
	src := backend.NewTensor(2, 2, []float32{
		1, 2,
		3, 4,
	})

	// Paste src into dst at position [1,1]
	dst.Paste(1, 1, src, 0, 0, 2, 2)

	result := dst.ToHost()

	// Expected:
	// 0, 0, 0
	// 0, 1, 2
	// 0, 3, 4
	expected := []float32{
		0, 0, 0,
		0, 1, 2,
		0, 3, 4,
	}

	for i, e := range expected {
		if math.Abs(float64(result[i]-e)) > 0.001 {
			t.Errorf("Paste: expected %v, got %v", expected, result)
			break
		}
	}
}
