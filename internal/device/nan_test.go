//go:build darwin && metal

package device

import (
	"math"
	"testing"
)

func TestMetal_HasNaN_Correctness(t *testing.T) {
	b := NewMetalBackendFP16()
	
	t.Run("Float32", func(t *testing.T) {
		data := []float32{1.0, float32(math.NaN()), 3.0}
		tm := b.NewTensorWithType(1, 3, Float32, data)
		hasNaN, err := tm.(*MetalTensor).HasNaN()
		if err != nil { t.Fatalf("HasNaN failed: %v", err) }
		if !hasNaN { t.Error("Expected true for FP32 NaN") }
		
		dataOk := []float32{1.0, 2.0, 3.0}
		tmOk := b.NewTensorWithType(1, 3, Float32, dataOk)
		hasNaN, _ = tmOk.(*MetalTensor).HasNaN()
		if hasNaN { t.Error("Expected false for FP32 OK") }
	})
	
	t.Run("Float16", func(t *testing.T) {
		data := []float32{1.0, float32(math.NaN()), 3.0}
		tm := b.NewTensorWithType(1, 3, Float16, data)
		hasNaN, err := tm.(*MetalTensor).HasNaN()
		if err != nil { t.Fatalf("HasNaN failed: %v", err) }
		if !hasNaN { t.Error("Expected true for FP16 NaN") }
		
		dataOk := []float32{1.0, 2.0, 3.0}
		tmOk := b.NewTensorWithType(1, 3, Float16, dataOk)
		hasNaN, _ = tmOk.(*MetalTensor).HasNaN()
		if hasNaN { t.Error("Expected false for FP16 OK") }
	})
}
