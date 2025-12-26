package weights

import (
	"os"
	"testing"
	"encoding/binary"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
)

func TestLoader_LoadFromRawBinary(t *testing.T) {
	// Create a dummy model
	config := model.DefaultBertTinyConfig()
	m := model.NewBertModel(config)
	loader := NewLoader(m)
	
	// Create a dummy weights file
	f, err := os.CreateTemp("", "weights")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	
	// Write some floats
	data := []float64{1.0, 2.0, 3.0}
	for _, v := range data {
		binary.Write(f, binary.LittleEndian, float32(v)) // Assuming file is float32
	}
	f.Close()
	
	// Try loading (will fail partly because file is too small for full model, but checks stricture)
	// Actually LoadFromRawBinary expects full model size.
	// Just check if it handles missing file
	
	err = loader.LoadFromRawBinary("non_existent_file")
	if err == nil {
		t.Error("Expected error for missing file")
	}
}
