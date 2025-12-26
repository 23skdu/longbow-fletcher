package embeddings

import (
	"testing"
)

func TestGenerateLorem(t *testing.T) {
	counts := []int{1, 5, 10}
	
	for _, count := range counts {
		texts := GenerateLorem(count)
		if len(texts) != count {
			t.Errorf("GenerateLorem(%d) returned %d texts", count, len(texts))
		}
		
		for _, text := range texts {
			if len(text) == 0 {
				t.Error("Generated empty lorem text")
			}
		}
	}
}

func TestGenerateLorem_Zero(t *testing.T) {
	texts := GenerateLorem(0)
	if len(texts) != 0 {
		t.Errorf("GenerateLorem(0) returned %d texts", len(texts))
	}
}
