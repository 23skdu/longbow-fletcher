package model

import "testing"

func FuzzModel(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
