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
	neverSplit     map[string]bool
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
		neverSplit: map[string]bool{
			"[UNK]": true, "[SEP]": true, "[PAD]": true, "[CLS]": true, "[MASK]": true,
		},
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

// isPunctuation checks if a rune is a punctuation character.
func isPunctuation(r rune) bool {
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

// splitOnPunctuation splits text on punctuation, keeping punctuation as separate tokens.
// It respects neverSplit tokens.
func (t *WordPieceTokenizer) splitOnPunctuation(text string) []string {
	var tokens []string
	
	// First convert to runes for safe iteration
	runes := []rune(text)
	var currentToken strings.Builder
	
	i := 0
	for i < len(runes) {
		// Check never_split patterns
		// Note: Checking string prefix on rune slice requires conversion or careful indexing.
		// For simplicity/correctness, we'll convert the suffix back to string.
		// Performance note: This is O(N^2) in worst case due to string creation, but text is short (512 chars).
		suffix := string(runes[i:])
		matched := false
		for ns := range t.neverSplit {
			if strings.HasPrefix(suffix, ns) {
				if currentToken.Len() > 0 {
					tokens = append(tokens, currentToken.String())
					currentToken.Reset()
				}
				tokens = append(tokens, ns)
				i += len([]rune(ns))
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		
		r := runes[i]
		if isPunctuation(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			// Check for whitespace to split words as well?
			// The original implementation relied on strings.Fields() later.
			// But here we are processing the whole string.
			// Standard BasicTokenizer splits on whitespace AND punctuation.
			// If we implement proper BasicTokenize here, we should split on whitespace too.
			if unicode.IsSpace(r) {
				if currentToken.Len() > 0 {
					tokens = append(tokens, currentToken.String())
					currentToken.Reset()
				}
				// Whitespace is eaten (not added as token)
			} else {
				currentToken.WriteRune(r)
			}
		}
		i++
	}
	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}
	return tokens
}

// Tokenize implement the WordPiece algorithm.
func (t *WordPieceTokenizer) Tokenize(text string) ([]string, []int) {
	// 1. Split on whitespace & punctuation, preserving special tokens
	rawTokens := t.splitOnPunctuation(text)
	
	outputTokens := make([]string, 0, len(rawTokens)*2)
	outputIDs := make([]int, 0, len(rawTokens)*2)

	for _, token := range rawTokens {
		// one token could be empty if multiple spaces?
		if token == "" { continue }
		
		// Check if special token
		if t.neverSplit[token] {
			if id, ok := t.vocab[token]; ok {
				outputTokens = append(outputTokens, token)
				outputIDs = append(outputIDs, id)
				continue
			}
		}
		
		// Normalization for regular tokens
		normToken := strings.ToLower(token)
		tform := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
		normToken, _, _ = transform.String(tform, normToken)
		
		// WordPiece
		if len(normToken) > t.maxInputChars {
			outputTokens = append(outputTokens, t.unkToken)
			outputIDs = append(outputIDs, t.vocab[t.unkToken])
			continue
		}

		isBad := false
		start := 0
		var subTokens []string
		for start < len(normToken) {
			end := len(normToken)
			var curSubstr string
			for start < end {
				substr := normToken[start:end]
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
