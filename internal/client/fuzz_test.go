package client

import "testing"

func FuzzClient(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
