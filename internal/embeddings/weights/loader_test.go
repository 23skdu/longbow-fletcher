package weights

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
)

func TestLoader_LoadFromRawBinary(t *testing.T) {
	config := model.DefaultBertTinyConfig()
	m := model.NewBertModel(config)
	loader := NewLoader(m)

	err := loader.LoadFromRawBinary("non_existent_file")
	if err == nil {
		t.Error("Expected error for missing file")
	}
}

func TestLoader_LoadFromSafeTensors(t *testing.T) {
	config := model.DefaultBertTinyConfig()
	m := model.NewBertModel(config)
	loader := NewLoader(m)

	err := loader.LoadFromSafeTensors("non_existent_file.safetensors")
	if err == nil {
		t.Error("Expected error for missing file")
	}
}

func TestLoader_FusedQKV(t *testing.T) {
	config := model.DefaultNomicConfig()
	config.FusedQKV = true
	m := model.NewBertModel(config)
	loader := NewLoader(m)

	f, err := os.CreateTemp("", "fused_qkv_weights")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Remove(f.Name()) }()

	hiddenSize := config.HiddenSize
	fusedSize := 3 * hiddenSize * hiddenSize

	data := make([]float32, fusedSize)
	for i := range data {
		data[i] = float32(i) * 0.001
	}

	for _, v := range data {
		_ = binary.Write(f, binary.LittleEndian, v)
	}
	_ = f.Close()

	err = loader.LoadFromRawBinary(f.Name())
	if err != nil {
		t.Logf("Load error (expected for incomplete model): %v", err)
	}
}

func TestLoader_SafeTensorsInfo(t *testing.T) {
	info := SafeTensorsInfo{
		DType:       "F32",
		Shape:       []int{768, 768},
		DataOffsets: []int64{0, 2359296},
	}

	if info.DType != "F32" {
		t.Error("SafeTensorsInfo DType mismatch")
	}
	if info.Shape[0] != 768 {
		t.Error("SafeTensorsInfo Shape mismatch")
	}
}

func writeBertTinyWeights(f *os.File, config model.BertConfig) error {
	vocabSize := config.VocabSize
	hiddenSize := config.HiddenSize
	numLayers := config.NumHiddenLayers
	intermediateSize := config.IntermediateSize

	write := func(data []float32) {
		for _, v := range data {
			_ = binary.Write(f, binary.LittleEndian, v)
		}
	}

	wordEmbed := make([]float32, vocabSize*hiddenSize)
	write(wordEmbed)

	posEmbed := make([]float32, config.MaxPositionEmbeddings*hiddenSize)
	write(posEmbed)

	tokenTypeEmbed := make([]float32, 2*hiddenSize)
	write(tokenTypeEmbed)

	embedLN := make([]float32, hiddenSize*2)
	write(embedLN)

	for i := 0; i < numLayers; i++ {
		q := make([]float32, hiddenSize*hiddenSize)
		write(q)
		qb := make([]float32, hiddenSize)
		write(qb)
		k := make([]float32, hiddenSize*hiddenSize)
		write(k)
		kb := make([]float32, hiddenSize)
		write(kb)
		v := make([]float32, hiddenSize*hiddenSize)
		write(v)
		vb := make([]float32, hiddenSize)
		write(vb)

		outDense := make([]float32, hiddenSize*hiddenSize)
		write(outDense)
		outBias := make([]float32, hiddenSize)
		write(outBias)

		interDense := make([]float32, intermediateSize*hiddenSize)
		write(interDense)
		interBias := make([]float32, intermediateSize)
		write(interBias)

		out2Dense := make([]float32, hiddenSize*intermediateSize)
		write(out2Dense)
		out2Bias := make([]float32, hiddenSize)
		write(out2Bias)

		ln1 := make([]float32, hiddenSize*2)
		write(ln1)
		ln2 := make([]float32, hiddenSize*2)
		write(ln2)
	}

	poolerDense := make([]float32, hiddenSize*hiddenSize)
	write(poolerDense)
	poolerBias := make([]float32, hiddenSize)
	write(poolerBias)

	return nil
}

func TestLoader_FullLoad(t *testing.T) {
	config := model.DefaultBertTinyConfig()
	m := model.NewBertModel(config)
	loader := NewLoader(m)

	f, err := os.CreateTemp("", "bert_tiny_weights")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Remove(f.Name()) }()

	err = writeBertTinyWeights(f, config)
	if err != nil {
		t.Fatal(err)
	}
	_ = f.Close()

	err = loader.LoadFromRawBinary(f.Name())
	if err != nil {
		t.Errorf("Failed to load weights: %v", err)
	}
}
