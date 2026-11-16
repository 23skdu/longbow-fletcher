package main

import "testing"

func FuzzFletcher(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
