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

	_, rawIDs := tok.Tokenize(*text)
	clsID := 101; sepID := 102
	inputIDs := append([]int{clsID}, rawIDs...)
	inputIDs = append(inputIDs, sepID)

	// 4. Run up to Attention Input (Embeddings Output)
	// We need exact same input tensor to verify Attention in isolation OR
	// we just rely on previous step verification and run full forward up to here.
	// Since Step 3 passed, we can rely on m.Embeddings.Forward.
	
	embeddingOutput := m.Embeddings.Forward(inputIDs)
	
	// 5. Run Attention Forward for Layer 0
	layer0 := m.Encoder.Layers[0]
	attn := layer0.Attention.Self
	
	// Q, K, V are internal now in Forward, but we can verify the output Context Layer
	// This exercises the new MHA loop in bert.go
	contextLayer := attn.Forward(embeddingOutput)
	
	dumps := []TensorDump{}
	dump := func(name string, t device.Tensor) {
		r, c := t.Dims()
		dumps = append(dumps, TensorDump{
			Name:   name,
			Values: t.ToHost(),
			Shape:  []int{r, c},
		})
	}

	dump("context_layer", contextLayer)

	// Output
	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(dumps); err != nil {
		log.Fatal(err)
	}
}
