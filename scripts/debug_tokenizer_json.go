package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
)

type TestCase struct {
	Text   string   `json:"text"`
	Tokens []string `json:"tokens"`
	IDs    []int    `json:"ids"`
}

type Output struct {
	Cases []TestCase `json:"cases"`
}

func main() {
	tok, err := tokenizer.NewWordPieceTokenizer("vocab.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load tokenizer: %v\n", err)
		os.Exit(1)
	}

	texts := []string{
		"Hello world",
		"hello world",
		"Hello World!",
		"The quick brown fox jumps over the lazy dog.",
		"unaffable", // likely subwords: un, ##aff, ##able
		"1234567890",
		"mixed case with 123 numbers",
		"[CLS] special tokens [SEP]",
		"café", // unicode
	}

	var output Output
	for _, text := range texts {
		tokens, ids := tok.Tokenize(text)
		output.Cases = append(output.Cases, TestCase{
			Text:   text,
			Tokens: tokens,
			IDs:    ids,
		})
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to encode output: %v\n", err)
		os.Exit(1)
	}
}
