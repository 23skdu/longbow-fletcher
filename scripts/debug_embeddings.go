package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
)

type TensorDump struct {
	Name   string    `json:"name"`
	Values []float32 `json:"values"`
	Shape  []int     `json:"shape"`
}

func main() {
	weightsPath := flag.String("weights", "bert_tiny.bin", "Path to weights binary")
	vocabPath := flag.String("vocab", "vocab.txt", "Path to vocab file")
	text := flag.String("text", "hello world", "Input text")
	flag.Parse()

	// 1. Setup
	backend := device.NewCPUBackend()
	config := model.DefaultBertTinyConfig()
	m := model.NewBertModelWithBackend(config, backend)

	// 2. Load Weights
	loader := weights.NewLoader(m)
	if err := loader.LoadFromRawBinary(*weightsPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// 3. Tokenize
	tok, err := tokenizer.NewWordPieceTokenizer(*vocabPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Manual Tokenization to match logic (Add [CLS] and [SEP])
	// In 'EmbedBatch', fletcher does:
	// tokens = [CLS] + tokens + [SEP] (if length allows)
	// We'll simulate a simple single sequence
	_, rawIDs := tok.Tokenize(*text)
	
	// Add CLS (101) and SEP (102) - verify these IDs in vocab.txt or assume standard BERT
	clsID := 101 // Standard BERT
	sepID := 102 // Standard BERT
	
	inputIDs := append([]int{clsID}, rawIDs...)
	inputIDs = append(inputIDs, sepID)
	
	// Print IDs for verification
	// fmt.Fprintf(os.Stderr, "Input IDs: %v\n", inputIDs)

	// 4. Run Embeddings Components Manually
	
	dumps := []TensorDump{}
	dump := func(name string, t device.Tensor) {
		r, c := t.Dims()
		dumps = append(dumps, TensorDump{
			Name:   name,
			Values: t.ToHost(),
			Shape:  []int{r, c},
		})
	}

	// A. Word Embeddings
	wordEmbeds := m.Embeddings.WordEmbeddings.Gather(inputIDs)
	dump("word_embeddings", wordEmbeds)

	// B. Position Embeddings
	seqLen := len(inputIDs)
	posIndices := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		posIndices[i] = i
	}
	posEmbeds := m.Embeddings.PositionEmbeddings.Gather(posIndices)
	dump("position_embeddings", posEmbeds)

	// C. Token Type Embeddings (All 0)
	typeIndices := make([]int, seqLen) // Zeros
	typeEmbeds := m.Embeddings.TokenTypeEmbeddings.Gather(typeIndices)
	dump("token_type_embeddings", typeEmbeds)

	// D. Sum (Word + Pos + Type)
	// Need to copy wordEmbeds because Add is in-place
	r, c := wordEmbeds.Dims()
	sumEmbeds := backend.NewTensor(r, c, nil)
	sumEmbeds.Copy(wordEmbeds)
	sumEmbeds.Add(posEmbeds)
	sumEmbeds.Add(typeEmbeds)
	dump("sum_embeddings", sumEmbeds)

	// E. LayerNorm + Dropout
	// Note: Fletcher's Dropout is Identity.
	// We need to copy again to avoid modifying 'sumEmbeds' in the dump if we were to reuse it, 
	// but here we just flow forward.
	lnEmbeds := backend.NewTensor(r, c, nil)
	lnEmbeds.Copy(sumEmbeds)
	m.Embeddings.LayerNorm.Forward(lnEmbeds)
	dump("layernorm_embeddings", lnEmbeds)

	// Output
	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(dumps); err != nil {
		log.Fatal(err)
	}
}
