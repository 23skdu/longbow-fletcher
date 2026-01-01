package weights

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/device"
)

// Loader handles loading model weights from binary files.
type Loader struct {
	Model *model.BertModel
}

// NewLoader creates a new weight loader for the given model.
func NewLoader(m *model.BertModel) *Loader {
	return &Loader{Model: m}
}

// LoadFromRawBinary loads weights from a raw binary file where Each matrix/bias 
// is stored as a sequence of float32 values (LittleEndian).
// This is a simplified implementation for the pure Go CLI.
func (l *Loader) LoadFromRawBinary(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()

	// In a real implementation, we would have a header or a manifest.
	// For this CLI, we'll assume a specific order based on the model structure.
	
	// Load Embeddings
	if err := l.loadDense(file, l.Model.Embeddings.WordEmbeddings); err != nil {
		return fmt.Errorf("failed to load word embeddings: %w", err)
	}
	if err := l.loadDense(file, l.Model.Embeddings.PositionEmbeddings); err != nil {
		return fmt.Errorf("failed to load position embeddings: %w", err)
	}
	if err := l.loadDense(file, l.Model.Embeddings.TokenTypeEmbeddings); err != nil {
		return fmt.Errorf("failed to load token type embeddings: %w", err)
	}
	if err := l.loadLayerNorm(file, l.Model.Embeddings.LayerNorm); err != nil {
		return fmt.Errorf("failed to load embedding layernorm: %w", err)
	}

	// Load Encoder Layers
	for i, layer := range l.Model.Encoder.Layers {
		// Attention
		if err := l.loadSelfAttention(file, layer.Attention.Self); err != nil {
			return fmt.Errorf("failed to load self-attention for layer %d: %w", i, err)
		}
		if err := l.loadSelfOutput(file, layer.Attention.Output); err != nil {
			return fmt.Errorf("failed to load attention output for layer %d: %w", i, err)
		}
		// Intermediate
		if err := l.loadDense(file, layer.Intermediate.Dense); err != nil {
			return fmt.Errorf("failed to load intermediate dense for layer %d: %w", i, err)
		}
		if err := l.loadDense(file, layer.Intermediate.Bias); err != nil {
			return fmt.Errorf("failed to load intermediate bias for layer %d: %w", i, err)
		}
		// Output
		if err := l.loadDense(file, layer.Output.Dense); err != nil {
			return fmt.Errorf("failed to load output dense for layer %d: %w", i, err)
		}
		if err := l.loadDense(file, layer.Output.Bias); err != nil {
			return fmt.Errorf("failed to load output bias for layer %d: %w", i, err)
		}
		if err := l.loadLayerNorm(file, layer.Output.LayerNorm); err != nil {
			return fmt.Errorf("failed to load output layernorm for layer %d: %w", i, err)
		}
	}

	// Load Pooler
	if err := l.loadDense(file, l.Model.Pooler.Dense); err != nil {
		return fmt.Errorf("failed to load pooler dense: %w", err)
	}
	if err := l.loadDense(file, l.Model.Pooler.Bias); err != nil {
		return fmt.Errorf("failed to load pooler bias: %w", err)
	}

	return nil
}

func (l *Loader) loadDense(r io.Reader, d device.Tensor) error {
	rows, cols := d.Dims()
	size := rows * cols
	data := make([]float32, size)
	
	// Read everything into a slice first (bulk read)
	// We assume model files are float32/LittleEndian
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return err
	}

	// Bulk upload to device
	d.CopyFromFloat32(data)
	return nil
}

func (l *Loader) loadLayerNorm(r io.Reader, ln *model.LayerNorm) error {
	if err := l.loadDense(r, ln.Gamma); err != nil {
		return err
	}
	return l.loadDense(r, ln.Beta)
}

func (l *Loader) loadSelfAttention(r io.Reader, sa *model.BertSelfAttention) error {
	if err := l.loadDense(r, sa.Query); err != nil {
		return err
	}
	if err := l.loadDense(r, sa.QueryBias); err != nil {
		return err
	}
	if err := l.loadDense(r, sa.Key); err != nil {
		return err
	}
	if err := l.loadDense(r, sa.KeyBias); err != nil {
		return err
	}
	if err := l.loadDense(r, sa.Value); err != nil {
		return err
	}
	if err := l.loadDense(r, sa.ValueBias); err != nil {
		return err
	}
	return nil
}

func (l *Loader) loadSelfOutput(r io.Reader, so *model.BertSelfOutput) error {
	if err := l.loadDense(r, so.Dense); err != nil {
		return err
	}
	if err := l.loadDense(r, so.Bias); err != nil {
		return err
	}
	return l.loadLayerNorm(r, so.LayerNorm)
}

// SafeTensorsInfo holds metadata for a single tensor in the SafeTensors header
type SafeTensorsInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets []int64  `json:"data_offsets"`
}

// LoadFromSafeTensors loads weights from a SafeTensors file.
func (l *Loader) LoadFromSafeTensors(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// 1. Read Header Size (uint64)
	var headerSize uint64
	if err := binary.Read(file, binary.LittleEndian, &headerSize); err != nil {
		return fmt.Errorf("failed to read header size: %w", err)
	}

	// 2. Read Header JSON
	if headerSize > 100*1024*1024 { // Sanity check: 100MB header limit
		return fmt.Errorf("header size too large: %d", headerSize)
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return fmt.Errorf("failed to read header: %w", err)
	}

	var header map[string]SafeTensorsInfo
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return fmt.Errorf("failed to parse header JSON: %w", err)
	}

    dataStart := int64(8) + int64(headerSize)

	// Helper to load by name
	load := func(name string, dest device.Tensor) error {
		info, ok := header[name]
		if !ok {
			return fmt.Errorf("tensor %q not found in SafeTensors file", name)
		}

		r, c := dest.Dims()
		expectedSize := int64(r * c)
		
		// Map logic: Check total size equality (allowing for transpose)
		var headerSize int64 = 1
		for _, dim := range info.Shape {
			headerSize *= int64(dim)
		}
		
		if headerSize != expectedSize {
             return fmt.Errorf("tensor %q size mismatch: expected %d, got %d (shape %v)", name, expectedSize, headerSize, info.Shape)
		}

		// Offset is relative to the START of the file? No, relative to end of header.
		// Spec: "The offsets are relative to the beginning of the body of the file, not the beginning of the file."
		// Body starts after 8 bytes (size) + headerSize.
        // dataStart is calculated outside.
		offsetStart := dataStart + info.DataOffsets[0]
		length := info.DataOffsets[1] - info.DataOffsets[0]
		if length != expectedSize * 4 { // Float32 = 4 bytes
             if info.DType == "F16" && length == expectedSize * 2 {
                 return fmt.Errorf("tensor %q is F16, only F32 supported currently", name)
             }
			return fmt.Errorf("tensor %q byte size mismatch: expected %d, got %d", name, expectedSize*4, length)
		}

		data := make([]byte, length)
		if _, err := file.ReadAt(data, offsetStart); err != nil {
			return fmt.Errorf("failed to read data for %q: %w", name, err)
		}

		// Cast to []float32
		floats := make([]float32, expectedSize)
        // Manual conversion LittleEndian
        for i := range floats {
            floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))
        }
        
        // Handle Transpose!
        // Heuristic: If name ends in ".weight" and it's 2D and not an Embedding or Norm.
        isLinearWeight := strings.HasSuffix(name, ".weight") && 
                          len(info.Shape) == 2 && 
                          !strings.Contains(name, "embeddings") && 
                          !strings.Contains(name, "LayerNorm")
                          
        if isLinearWeight {
            outRows := info.Shape[0]
            inCols := info.Shape[1]
            if outRows != c || inCols != r {
                 return fmt.Errorf("tensor %q shape mismatch for transpose: header=%v, expected (%d, %d)", name, info.Shape, r, c)
            }
            
            transposed := make([]float32, expectedSize)
            for i := 0; i < outRows; i++ {
                for j := 0; j < inCols; j++ {
                    // Original (i, j) -> New (j, i)
                    transposed[j*outRows + i] = floats[i*inCols + j]
                }
            }
            floats = transposed
        }


		dest.CopyFromFloat32(floats)
		return nil
	}

	// Load Embeddings
	if err := load("embeddings.word_embeddings.weight", l.Model.Embeddings.WordEmbeddings); err != nil { return err }
	if err := load("embeddings.position_embeddings.weight", l.Model.Embeddings.PositionEmbeddings); err != nil { return err }
	if err := load("embeddings.token_type_embeddings.weight", l.Model.Embeddings.TokenTypeEmbeddings); err != nil { return err }
	if err := load("embeddings.LayerNorm.weight", l.Model.Embeddings.LayerNorm.Gamma); err != nil { return err }
	if err := load("embeddings.LayerNorm.bias", l.Model.Embeddings.LayerNorm.Beta); err != nil { return err }

	// Load Encoder Layers
	for i, layer := range l.Model.Encoder.Layers {
		prefix := fmt.Sprintf("encoder.layer.%d", i)
		// Attention
		if err := load(prefix+".attention.self.query.weight", layer.Attention.Self.Query); err != nil { return err }
		if err := load(prefix+".attention.self.query.bias", layer.Attention.Self.QueryBias); err != nil { return err }
		if err := load(prefix+".attention.self.key.weight", layer.Attention.Self.Key); err != nil { return err }
		if err := load(prefix+".attention.self.key.bias", layer.Attention.Self.KeyBias); err != nil { return err }
		if err := load(prefix+".attention.self.value.weight", layer.Attention.Self.Value); err != nil { return err }
		if err := load(prefix+".attention.self.value.bias", layer.Attention.Self.ValueBias); err != nil { return err }
		if err := load(prefix+".attention.output.dense.weight", layer.Attention.Output.Dense); err != nil { return err }
		if err := load(prefix+".attention.output.dense.bias", layer.Attention.Output.Bias); err != nil { return err }
		if err := load(prefix+".attention.output.LayerNorm.weight", layer.Attention.Output.LayerNorm.Gamma); err != nil { return err }
		if err := load(prefix+".attention.output.LayerNorm.bias", layer.Attention.Output.LayerNorm.Beta); err != nil { return err }
		
		// Intermediate
		if err := load(prefix+".intermediate.dense.weight", layer.Intermediate.Dense); err != nil { return err }
		if err := load(prefix+".intermediate.dense.bias", layer.Intermediate.Bias); err != nil { return err }
		
		// Output
		if err := load(prefix+".output.dense.weight", layer.Output.Dense); err != nil { return err }
		if err := load(prefix+".output.dense.bias", layer.Output.Bias); err != nil { return err }
		if err := load(prefix+".output.LayerNorm.weight", layer.Output.LayerNorm.Gamma); err != nil { return err }
		if err := load(prefix+".output.LayerNorm.bias", layer.Output.LayerNorm.Beta); err != nil { return err }
	}

	// Load Pooler
	if err := load("pooler.dense.weight", l.Model.Pooler.Dense); err != nil { return err }
	if err := load("pooler.dense.bias", l.Model.Pooler.Bias); err != nil { return err }

	return nil
}
