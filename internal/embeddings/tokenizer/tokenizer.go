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
	// Optimization: If text is pure ASCII, use the SIMD/Lookup optimized finder.
	isASCII := true
	for i := 0; i < len(text); i++ {
		if text[i] >= 128 {
			isASCII = false
			break
		}
	}

	if isASCII {
		return t.splitOnPunctuationASCII(text)
	}

	// Fallback to Rune-based iteration for Unicode
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

// splitOnPunctuationASCII is the optimized path for ASCII text.
func (t *WordPieceTokenizer) splitOnPunctuationASCII(text string) []string {
	var tokens []string
	start := 0
	
	for i := 0; i < len(text); {
		// Check never_split patterns first
		// Since we are ASCII, slicing is cheap O(1) inside loop (no rune conversion)
		suffix := text[i:]
		matched := false
		for ns := range t.neverSplit {
			if strings.HasPrefix(suffix, ns) {
				if i > start {
					tokens = append(tokens, text[start:i])
				}
				tokens = append(tokens, ns)
				i += len(ns)
				start = i
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		// Use FindPunctuation to find next separator efficiently
		// Note: FindPunctuation scans bytes. It returns offset relative to slice.
		offset := FindPunctuation([]byte(suffix))
		if offset == -1 {
			// No more separators (punct or space)
			// Consume rest
			i = len(text)
			break
		}
		
		// Found a separator at i + offset
		sepIdx := i + offset
		if sepIdx > start {
			// We skipped some previous text (tokens)
			// We need to handle them?
			// Wait. FindPunctuation jumps to *first* separator.
			// So text[start:sepIdx] is a token (if valid).
			// But we must account for the fact that we might have skipped `neverSplit` checks?
			// If `FindPunctuation` jumps 10 chars.
			// Could there be a `[UNK]` in there?
			// Yes.
			// So we cannot assume text[start:sepIdx] is clean if `neverSplit` exists.
			// But we checked `neverSplit` at `i` (start of suffix).
			// If `FindPunctuation` jumps, we might skip a `neverSplit` at `i+1`.
			// So we can only use `FindPunctuation` if `sepIdx` is `i` (immediate match).
			// If `offset > 0`?
			// We can verify `neverSplit` rapidly?
			// Or only use `FindPunctuation` to identify if *current* char is separator?
			// That defeats SIMD speed (scanning).
			// However, `neverSplit` tokens are Rare.
			// Most text is words.
			// Checking `neverSplit` at every byte is slow.
			// But `FindPunctuation` doesn't know about `neverSplit`.
			//
			// If `neverSplit` tokens always start with Punctuation (e.g. '[').
			// `FindPunctuation` WILL stop at '['.
			// So we are safe provided all `neverSplit` map keys start with a char found by `FindPunctuation`.
			// `[UNK]`, `[CLS]` -> Start with `[`. `[` is punctuation. Found.
			// So `FindPunctuation` will stop at start of special tokens.
			// What if special token starts with non-punct? e.g. "MASK"? (Usually it is "[MASK]").
			// If user defines custom neverSplit "foo"...
			// If "foo" starts with 'f' (non-punct).
			// `FindPunctuation` skips 'f'. We miss it.
			// Assumption: neverSplit tokens start with punctuation or we take the hit.
			// Given standard BERT vocab, this holds.
			
			// Correct logic using FindPunctuation:
			// 1. Scan for separator (Punct or Space) starting at `i`.
			// 2. Found at `sepIdx`.
			// 3. Before accepting `sepIdx`, we must check if any `neverSplit` occurs in `text[i:sepIdx]`.
			//    This implies checking every position `j` in range. Slow.
			//    BUT if we assume `neverSplit` start chars are caught by `FindPunctuation`.
			//    Then `sepIdx` IS the location of potential `neverSplit`.
			//    So we process text up to `sepIdx` as a word.
			//    Then process `text[sepIdx]` (separator).
			
			// So: 
			// 1. `sepIdx = i + offset`.
			// 2. Token `text[start:sepIdx]` (if len>0) -> Add.
			// 3. `i = sepIdx`.
			// 4. Check `neverSplit` at `i`. (Handled by main loop top).
			// 5. If not `neverSplit`, process `text[i]` as Punct/Space.
			//6. `i++`.
			
			// Wait. The main loop checks `neverSplit` then does logic.
			// If I reuse main loop structure but advance `i` using `FindPunctuation`?
			// `i` points to current char.
			// Check `neverSplit` at `i`.
			// If match -> handled.
			// If no match ->
			//    Check if `text[i]` is separator?
			//    If NO -> Logic is "Consume char".
			//    In scalar loop, "Consume char" means `currToken.Write`.
			//    In optimized loop, we want to Skip Many Chars.
			//    `offset = FindPunctuation(text[i:])`.
			//    If offset > 0:
			//       We found a chunk `text[i : i+offset]` of NON-separators.
			//       Does this chunk contain `neverSplit`?
			//       Only if `neverSplit` starts with non-separator.
			//       If we assume they start with `[`, `FindPunctuation` STOPS at `[` (index 91).
			//       So `FindPunctuation` will NOT skip over `[`.
			//       So `offset` points strictly to the next "Event".
			//       So we can safely append `text[i : i+offset]` to tokens (or currToken buffer).
			//       So `tokens = append(tokens, text[start : i+offset])`? No.
			//       We accumulate token pieces.
			//       Actually `start` tracks token start.
			//       If we skip `offset`, `start` stays.
			//       `i += offset`.
			//       Now `text[i]` is a Separator (Punct/Space) OR `neverSplit` logic will catch it next iter.
			//       So we `continue` main loop.
			
			// This logic is sound assuming `neverSplit` start chars are flagged by `FindPunctuation`.
			// List: `[` (91). Flagged.
			// If vocab has others?
			// We can conservatively assume yes.
			
			// Logic:
			// check neverSplit.
			// offset = FindPunctuation(text[i:]).
			// if offset > 0:
			//    i += offset (Skip non-separators).
			//    continue (This loops back to neverSplit check at new `i`).
			// if offset == 0: (Current char is separator).
			//    Handle separator.
			
			// Wait. `FindPunctuation` finds the *index* of match.
			// If current char is Match, `offset` is 0.
			// So we fall through to "Handle separator".
			// If current char is 'H', and next is ' ', `offset` is 1. 'e', 'l', 'l', 'o', ' '. `offset` is 5.
			// We advance `i` by 5. `i` is now at ' '.
			// Loop continues. `start` was 0.
			// Next iter: `i` at ' '. `neverSplit`? No.
			// `FindPunctuation` returns 0.
			// Fallthrough. Handle `text[i]` (' ').
			// Space -> Split. Token = `text[start:i]` ("Hello"). Add. Reset start.
			
			// This works!
			// Only issue: `FindPunctuation` must include `[` (91). Yes.
			
			offset := FindPunctuation([]byte(text[i:]))
			if offset > 0 {
				i += offset
				// Loop back to check neverSplit at new `i`
				continue
			}
			
			// At this point, `text[i]` is a separator (or FindPunctuation failed / -1 handled below).
			if offset == -1 {
				// Remainder is word.
				i = len(text)
				continue // Loop ends
			}
			
			// offset == 0. `text[i]` is Punct/Space.
			b := text[i]
			isSpace := (b == 32 || (b >= 9 && b <= 13))
			
			// If we have pending token
			if i > start {
				tokens = append(tokens, text[start:i])
			}
			
			if !isSpace {
				tokens = append(tokens, string(b))
			}
			
			i++
			start = i
		}
	}
	if start < len(text) {
		tokens = append(tokens, text[start:])
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
