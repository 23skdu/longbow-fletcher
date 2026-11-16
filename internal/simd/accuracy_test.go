package simd

import (
	"math"
	"testing"
)

func TestSIMDAccuracy(t *testing.T) {
	// ExpFast accuracy
	for _, x := range []float32{-2, -0.5, 0, 0.5, 2} {
		got := ExpFast(x)
		want := float32(math.Exp(float64(x)))
		if math.Abs(float64(got-want)) > 5e-2 {
			t.Errorf("ExpFast(%v) = %v, want %v", x, got, want)
		}
	}
	// TanhFast accuracy
	for _, x := range []float32{-3, -1, 0, 1, 3} {
		got := TanhFast(x)
		want := float32(math.Tanh(float64(x)))
		if math.Abs(float64(got-want)) > 5e-2 {
			t.Errorf("TanhFast(%v) = %v, want %v", x, got, want)
		}
	}
	// VecAdd
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	VecAdd(a, b)
	for i, v := range []float32{5, 7, 9} {
		if a[i] != v {
			t.Errorf("VecAdd result %v", a)
		}
	}
}

func TestOracleParity(t *testing.T) {
	// Ground-truth parity for SoftmaxFast sums to 1.
	row := []float32{1, 2, 3}
	SoftmaxFast(row)
	sum := float32(0)
	for _, v := range row {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Errorf("softmax sum = %v", sum)
	}
}
