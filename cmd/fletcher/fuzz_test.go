package main

import "testing"

func FuzzFletcher(f *testing.F) {
	f.Add("x")
	f.Fuzz(func(t *testing.T, s string) { _ = s })
}

func FuzzFlightEmbeddingBatch(f *testing.F) {
	f.Add(uint64(1), 2)
	f.Fuzz(func(t *testing.T, id uint64, dim int) {
		if dim < 1 || dim > 64 {
			return
		}
		ids := []uint64{id}
		vecs := make([]float32, dim)
		srv := NewFletcherFlightServer(nil)
		_ = srv.buildEmbeddingBatch(ids, vecs, dim)
	})
}
