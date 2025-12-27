package tokenizer

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTokenizer(t *testing.T) {
	// Create a dummy vocab.txt
	vocabContent := []string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"hello", "world", "hi", "how", "are", "you",
		"##lo", "##ld", "##i",
	}
	vocabPath := "test_vocab.txt"
	err := os.WriteFile(vocabPath, []byte(""), 0644)
	require.NoError(t, err)
	defer func() { _ = os.Remove(vocabPath) }()

	f, err := os.OpenFile(vocabPath, os.O_APPEND|os.O_WRONLY, 0644)
	require.NoError(t, err)
	for _, v := range vocabContent {
		_, _ = f.WriteString(v + "\n")
	}
	_ = f.Close()

	tk, err := NewWordPieceTokenizer(vocabPath)
	require.NoError(t, err)

	t.Run("BasicTokenize", func(t *testing.T) {
		tokens, ids := tk.Tokenize("Hello world")
		require.Equal(t, []string{"hello", "world"}, tokens)
		require.Equal(t, []int{5, 6}, ids)
	})

	t.Run("WordPieceSplit", func(t *testing.T) {
		tokens, ids := tk.Tokenize("hellold")
		require.Equal(t, []string{"hello", "##ld"}, tokens)
		require.Equal(t, []int{5, 12}, ids)
	})

	t.Run("UNKHandling", func(t *testing.T) {
		tokens, ids := tk.Tokenize("unknownword")
		require.Equal(t, []string{"[UNK]"}, tokens)
		require.Equal(t, []int{1}, ids)
	})

	t.Run("Normalization", func(t *testing.T) {
		tokens, ids := tk.Tokenize("Héllo")
		require.Equal(t, []string{"hello"}, tokens)
		require.Equal(t, []int{5}, ids)
	})
}
