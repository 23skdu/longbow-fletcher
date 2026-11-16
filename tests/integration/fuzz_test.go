package integration

import "testing"

func FuzzIntegration(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
