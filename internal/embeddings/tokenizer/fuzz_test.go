package tokenizer

import "testing"

func FuzzTokenizer(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
