package weights

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

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
