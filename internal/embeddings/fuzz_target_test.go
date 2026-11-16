package embeddings

import "testing"

func FuzzEmbeddings(f *testing.F) {
	f.Add("hello")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}
