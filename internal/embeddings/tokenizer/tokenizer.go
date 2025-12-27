package tokenizer

import (
	"bufio"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// Tokenizer defines the interface for text tokenization.
type Tokenizer interface {
	Tokenize(text string) ([]string, []int)
	Encode(text string) []int
}

// WordPieceTokenizer implements the WordPiece tokenization algorithm.
type WordPieceTokenizer struct {
	vocab          map[string]int
	invVocab       map[int]string
	maxInputChars  int
	unkToken       string
}

// NewWordPieceTokenizer creates a new WordPieceTokenizer from a vocab file.
func NewWordPieceTokenizer(vocabPath string) (*WordPieceTokenizer, error) {
	vocab, err := loadVocab(vocabPath)
	if err != nil {
		return nil, err
	}

	invVocab := make(map[int]string, len(vocab))
	for k, v := range vocab {
		invVocab[v] = k
	}

	return &WordPieceTokenizer{
		vocab:         vocab,
		invVocab:      invVocab,
		maxInputChars: 200,
		unkToken:      "[UNK]",
	}, nil
}

// loadVocab reads a BERT-style vocab.txt file.
func loadVocab(path string) (map[string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	vocab := make(map[string]int)
	scanner := bufio.NewScanner(file)
	index := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			vocab[line] = index
			index++
		}
	}
	return vocab, scanner.Err()
}

// BasicNormalize performs basic text normalization (lowercase, stripping accents).
func (t *WordPieceTokenizer) BasicNormalize(text string) string {
	text = strings.ToLower(text)
	
	tform := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	normText, _, _ := transform.String(tform, text)
	
	return normText
}

// Tokenize implement the WordPiece algorithm.
func (t *WordPieceTokenizer) Tokenize(text string) ([]string, []int) {
	text = t.BasicNormalize(text)
	words := strings.Fields(text)
	
	var outputTokens []string
	var outputIDs []int

	for _, word := range words {
		if len(word) > t.maxInputChars {
			outputTokens = append(outputTokens, t.unkToken)
			outputIDs = append(outputIDs, t.vocab[t.unkToken])
			continue
		}

		isBad := false
		start := 0
		var subTokens []string
		for start < len(word) {
			end := len(word)
			var curSubstr string
			for start < end {
				substr := word[start:end]
				if start > 0 {
					substr = "##" + substr
				}
				if _, ok := t.vocab[substr]; ok {
					curSubstr = substr
					break
				}
				end--
			}
			if curSubstr == "" {
				isBad = true
				break
			}
			subTokens = append(subTokens, curSubstr)
			start = end
		}

		if isBad {
			outputTokens = append(outputTokens, t.unkToken)
			outputIDs = append(outputIDs, t.vocab[t.unkToken])
		} else {
			for _, st := range subTokens {
				outputTokens = append(outputTokens, st)
				outputIDs = append(outputIDs, t.vocab[st])
			}
		}
	}

	return outputTokens, outputIDs
}

// Encode converts text into a slice of input IDs.
func (t *WordPieceTokenizer) Encode(text string) []int {
	_, ids := t.Tokenize(text)
	return ids
}
