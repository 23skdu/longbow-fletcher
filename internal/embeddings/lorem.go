package embeddings

import (
	"math/rand"
	"strings"
	"time"
)

var loremWords = []string{
	"lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
	"sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
	"magna", "aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud",
	"exercitation", "ullamco", "laboris", "nisi", "ut", "aliquip", "ex", "ea",
	"commodo", "consequat", "duis", "aute", "irure", "dolor", "in", "reprehenderit",
	"in", "voluptate", "velit", "esse", "cillum", "dolore", "eu", "fugiat", "nulla",
	"pariatur", "excepteur", "sint", "occaecat", "cupidatat", "non", "proident",
	"sunt", "in", "culpa", "qui", "officia", "deserunt", "mollit", "anim", "id", "est", "laborum",
}

// GenerateLorem generates a configurable number of Lorem Ipsum paragraphs.
func GenerateLorem(paragraphs int) []string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	result := make([]string, paragraphs)

	for i := 0; i < paragraphs; i++ {
		sentences := 3 + r.Intn(5)
		para := make([]string, sentences)
		for j := 0; j < sentences; j++ {
			wordCount := 5 + r.Intn(10)
			sentence := make([]string, wordCount)
			for k := 0; k < wordCount; k++ {
				sentence[k] = loremWords[r.Intn(len(loremWords))]
			}
			// Capitalize first word
			sentence[0] = strings.Title(sentence[0])
			para[j] = strings.Join(sentence, " ") + "."
		}
		result[i] = strings.Join(para, " ")
	}

	return result
}
