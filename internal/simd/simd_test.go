package simd

import (
	"math"
	"testing"
)

func TestVecAdd(t *testing.T) {
	dst := []float32{1, 2, 3, 4, 5}
	src := []float32{10, 20, 30, 40, 50}
	expected := []float32{11, 22, 33, 44, 55}

	VecAdd(dst, src)

	for i, v := range dst {
		if v != expected[i] {
			t.Errorf("VecAdd(%d) = %f, want %f", i, v, expected[i])
		}
	}
}

func TestVecAddScaled(t *testing.T) {
	dst := []float32{1, 2, 3, 4, 5}
	src := []float32{10, 20, 30, 40, 50}
	scale := float32(0.5)
	expected := []float32{6, 12, 18, 24, 30}

	VecAddScaled(dst, src, scale)

	for i, v := range dst {
		if v != expected[i] {
			t.Errorf("VecAddScaled(%d) = %f, want %f", i, v, expected[i])
		}
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{2, 3, 4, 5, 6}
	// 2 + 6 + 12 + 20 + 30 = 70
	expected := float32(70.0)

	result := DotProduct(a, b)

	if result != expected {
		t.Errorf("DotProduct = %f, want %f", result, expected)
	}
}

func TestMatVecMul(t *testing.T) {
	// 2x3 matrix
	mat := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	vec := []float32{1, 2, 3}
	dst := make([]float32, 2)
	
	// Row 0: 1*1 + 2*2 + 3*3 = 1+4+9 = 14
	// Row 1: 4*1 + 5*2 + 6*3 = 4+10+18 = 32
	expected := []float32{14, 32}

	MatVecMul(dst, mat, vec, 2, 3)

	for i, v := range dst {
		if v != expected[i] {
			t.Errorf("MatVecMul(%d) = %f, want %f", i, v, expected[i])
		}
	}
}

func TestFastMath(t *testing.T) {
	tests := []struct {
		name string
		fn   func(float32) float32
		std  func(float64) float64
		tol  float64 // Relative tolerance
	}{
		{"ExpFast", ExpFast, math.Exp, 0.05} , // 5% tolerance for approx
		{"TanhFast", TanhFast, math.Tanh, 0.05}, // 5% tolerance
	}

	inputs := []float32{-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, x := range inputs {
				got := tt.fn(x)
				want := tt.std(float64(x))
				
				diff := math.Abs(float64(got) - want)
				avg := math.Abs(want)
				if avg == 0 { avg = 1 }
				relErr := diff / avg

				// Relax tolerance for very small numbers where diff dominates
				if diff > 0.001 && relErr > tt.tol {
					t.Errorf("%s(%f) = %f, want %f (diff %f, rel %f)", tt.name, x, got, want, diff, relErr)
				}
			}
		})
	}
}

// Benchmarks

func BenchmarkDotProduct(b *testing.B) {
	size := 128
	v1 := make([]float32, size)
	v2 := make([]float32, size)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(v1, v2)
	}
}

func BenchmarkVecAdd(b *testing.B) {
	size := 128
	v1 := make([]float32, size)
	v2 := make([]float32, size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VecAdd(v1, v2)
	}
}

func BenchmarkExpFast(b *testing.B) {
	x := float32(0.5)
	for i := 0; i < b.N; i++ {
		ExpFast(x)
	}
}

func BenchmarkExpStd(b *testing.B) {
	x := 0.5
	for i := 0; i < b.N; i++ {
		math.Exp(x)
	}
}

func BenchmarkTanhFast(b *testing.B) {
	x := float32(0.5)
	for i := 0; i < b.N; i++ {
		TanhFast(x)
	}
}

func BenchmarkTanhStd(b *testing.B) {
	x := 0.5
	for i := 0; i < b.N; i++ {
		math.Tanh(x)
	}
}
