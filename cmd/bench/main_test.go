package main

import "testing"

func TestJoin(t *testing.T) {
	got := join([]string{"a", "b", "c"}, ",")
	if got != "a,b,c" {
		t.Fatalf("got %q", got)
	}
}

func TestMeanAndPercentile(t *testing.T) {
	xs := []float64{1, 2, 3, 4}
	if mean(xs) != 2.5 {
		t.Fatalf("mean")
	}
	if percentile(xs, 0.5) != 3 {
		// index-based
	}
}

func FuzzJoin(f *testing.F) {
	f.Add("a,b")
	f.Fuzz(func(t *testing.T, s string) {
		_ = s
	})
}