package device

import (
	"math"
	"testing"
)

func TestCPUBackend_TensorOps(t *testing.T) {
	backend := NewCPUBackend()

	t.Run("Add", func(t *testing.T) {
		a := backend.NewTensor(2, 2, []float64{1, 2, 3, 4})
		b := backend.NewTensor(2, 2, []float64{10, 20, 30, 40})
		
		a.Add(b)
		
		expected := []float64{11, 22, 33, 44}
		data := a.ToHost()
		
		for i, v := range expected {
			if math.Abs(data[i]-v) > 1e-6 {
				t.Errorf("Add mismatch at %d: got %f, want %f", i, data[i], v)
			}
		}
	})

	t.Run("Mul", func(t *testing.T) {
		// A: 2x3, B: 3x2 -> C: 2x2
		a := backend.NewTensor(2, 3, []float64{
			1, 2, 3,
			4, 5, 6,
		})
		b := backend.NewTensor(3, 2, []float64{
			7, 8,
			9, 10,
			11, 12,
		})
		
		c := backend.NewTensor(2, 2, nil)
		c.Mul(a, b)
		
		// 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
		// 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
		// 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
		// 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
		expected := []float64{58, 64, 139, 154}
		data := c.ToHost()
		
		for i, v := range expected {
			if math.Abs(data[i]-v) > 1e-6 {
				t.Errorf("Mul mismatch at %d: got %f, want %f", i, data[i], v)
			}
		}
	})

	t.Run("Scale", func(t *testing.T) {
		a := backend.NewTensor(2, 2, []float64{1, 2, 3, 4})
		a.Scale(2.0)
		
		expected := []float64{2, 4, 6, 8}
		data := a.ToHost()
		for i, v := range expected {
			if math.Abs(data[i]-v) > 1e-6 {
				t.Errorf("Scale mismatch at %d: got %f, want %f", i, data[i], v)
			}
		}
	})
	
	t.Run("LayerNorm", func(t *testing.T) {
		// 1x4 vector
		a := backend.NewTensor(1, 4, []float64{1, 2, 3, 4})
		gamma := backend.NewTensor(1, 4, []float64{1, 1, 1, 1})
		beta := backend.NewTensor(1, 4, []float64{0, 0, 0, 0})
		
		// Mean = 2.5
		// Variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
		// = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5 / 4 = 1.25
		// StdDev = sqrt(1.25) ≈ 1.11803
		
		// Expected: (val - 2.5) / 1.11803
		// 1: -1.5 / 1.11803 ≈ -1.34164
		// 2: -0.5 / 1.11803 ≈ -0.44721
		// 3: 0.5 / 1.11803 ≈ 0.44721
		// 4: 1.5 / 1.11803 ≈ 1.34164
		
		a.LayerNorm(gamma, beta, 1e-12)
		
		expected := []float64{-1.3416407, -0.4472136, 0.4472136, 1.3416407}
		data := a.ToHost()
		
		for i, v := range expected {
			if math.Abs(data[i]-v) > 1e-5 {
				t.Errorf("LayerNorm mismatch at %d: got %f, want %f", i, data[i], v)
			}
		}
	})
	
	t.Run("Pooling", func(t *testing.T) {
		t1 := backend.GetTensor(10, 10)
		t1.Set(0, 0, 123)
		backend.PutTensor(t1)
		
		t2 := backend.GetTensor(10, 10)
		// Should overwrite t1's memory, verify it is zeroed
		if val := t2.At(0, 0); val != 0 {
			t.Errorf("Pooled tensor not zeroed: got %f", val)
		}
		
		// Check pointer identity if we could, but we can't easily.
		// Trust implementation or check memory address via reflection (unsafe).
		// For now functonal test is enough.
	})
}
