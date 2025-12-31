package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
)

// WeightDump holds the summary of a loaded tensor for verification
type WeightDump struct {
	Name      string    `json:"name"`
	Rows      int       `json:"rows"`
	Cols      int       `json:"cols"`
	FirstFew  []float32 `json:"first_few"`
	LastFew   []float32 `json:"last_few"`
	Sum       float32   `json:"sum"`
}

func main() {
	modelType := flag.String("model", "bert-tiny", "Model type")
	weightsPath := flag.String("weights", "bert_tiny.bin", "Path to weights binary")
	flag.Parse()

	// 1. Create a CPU Backend
	backend := device.NewCPUBackend()

	// 2. Load Config
	var config model.BertConfig
	switch *modelType {
	case "bert-tiny":
		config = model.DefaultBertTinyConfig()
	case "nomic-embed-text":
		config = model.DefaultNomicConfig()
	default:
		log.Fatalf("Unknown model type: %s", *modelType)
	}

	// 3. Initialize Model (creates tensors)
	m := model.NewBertModelWithBackend(config, backend)

	// 4. Load Weights
	loader := weights.NewLoader(m)
	if err := loader.LoadFromRawBinary(*weightsPath); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// 5. Inspect specific tensors
	
	dumps := []WeightDump{}

	// Helper to dump a tensor
	dump := func(name string, t device.Tensor) {
		r, c := t.Dims()
		data := t.ToHost() // CPU backend returns slice copy
		
		wd := WeightDump{
			Name: name,
			Rows: r,
			Cols: c,
		}

		if len(data) > 0 {
			count := 5
			if len(data) < 5 { count = len(data) }
			wd.FirstFew = data[:count]
			wd.LastFew = data[len(data)-count:]
			
			sum := float32(0)
			for _, v := range data {
				sum += v
			}
			wd.Sum = sum
		}
		dumps = append(dumps, wd)
	}

	dump("embeddings.word_embeddings", m.Embeddings.WordEmbeddings)
	dump("embeddings.position_embeddings", m.Embeddings.PositionEmbeddings)
	dump("embeddings.token_type_embeddings", m.Embeddings.TokenTypeEmbeddings)
	dump("embeddings.LayerNorm.weight", m.Embeddings.LayerNorm.Gamma)
	dump("embeddings.LayerNorm.bias", m.Embeddings.LayerNorm.Beta)

	// Check Layer 0 Attention
	layer0 := m.Encoder.Layers[0]
	dump("encoder.layer.0.attention.self.query.weight", layer0.Attention.Self.Query)
	dump("encoder.layer.0.attention.self.query.bias", layer0.Attention.Self.QueryBias)
	dump("encoder.layer.0.attention.output.dense.weight", layer0.Attention.Output.Dense)

	// Pooler
	dump("pooler.dense.weight", m.Pooler.Dense)

	// Output JSON
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(dumps); err != nil {
		log.Fatal(err)
	}
}
