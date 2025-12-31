package tokenizer

var punctuationTable [256]bool

func init() {
	// Punctuation is defined as:
	// [33, 47] - ! " # $ % & ' ( ) * + , - . /
	// [58, 64] - : ; < = > ? @
	// [91, 96] - [ \ ] ^ _ `
	// [123, 126] - { | } ~
	// Plus whitespace: 9 (\t), 10 (\n), 11 (\v), 12 (\f), 13 (\r), 32 (space)
	for i := 0; i < 256; i++ {
		if (i >= 33 && i <= 47) || (i >= 58 && i <= 64) || (i >= 91 && i <= 96) || (i >= 123 && i <= 126) {
			punctuationTable[i] = true
		}
		if i == 32 || (i >= 9 && i <= 13) {
			punctuationTable[i] = true
		}
	}
}

// FindPunctuation scans the input byte slice for the first occurrence of a punctuation character.
// It uses a lookup table for efficient checking.
func FindPunctuation(text []byte) int {
	for i, b := range text {
		if punctuationTable[b] {
			return i
		}
	}
	return -1
}
