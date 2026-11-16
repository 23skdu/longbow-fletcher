package simd

import "testing"

func FuzzSIMD(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
