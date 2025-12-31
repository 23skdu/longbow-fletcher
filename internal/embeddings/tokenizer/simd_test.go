package tokenizer

import (
	"strings"
	"testing"
)

func TestFindPunctuation(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantIndex int
	}{
		{"Empty", "", -1},
		// "Hello world..." -> Space at 5.
		{"NoPunctuation", "Hello world this is a test", 5},
		{"StartPunctuation", "!Hello", 0},
		{"EndPunctuation", "Hello!", 5}, // ! is 5
		{"MiddlePunctuation", "Hello, world", 5}, // , is 5
		{"MultiplePunctuation", "Hello, world!", 5}, // , is 5
		{"Brackets", "[CLS]", 0},
		// "The quick..." -> Space at 3.
		{"ComplexChars", "The quick-brown fox.", 3},
		{"LongStringNoPunct", strings.Repeat("a", 100), -1},
		{"LongStringWithPunctAtEnd", strings.Repeat("a", 100) + "!", 100},
		{"MixedPunctuation", "#$%", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FindPunctuation([]byte(tt.input))
			if got != tt.wantIndex {
				t.Errorf("FindPunctuation(%q) = %d, want %d", tt.input, got, tt.wantIndex)
			}
		})
	}
}

func BenchmarkFindPunctuation(b *testing.B) {
	// Baseline benchmark to measure speedup later
	input := []byte(strings.Repeat("a", 64) + "!") // 65 chars
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FindPunctuation(input)
	}
}
