//go:build darwin && metal

package device

import (
	"encoding/binary"
	"testing"

)

func TestMetal_Cast_ExtractBytes(t *testing.T) {
	backend := NewMetalBackend() // Default FP32
	
	// Data: 1.0, -2.0
	// FP32: 0x3f800000, 0xc0000000
	// FP16: 0x3c00, 0xc000
	data := []float32{1.0, -2.0}
	tensor := backend.NewTensor(1, 2, data)
	
	// 1. Cast to FP16
	t16 := tensor.Cast(Float16)
	
	// 2. Extract Bytes
	bytes := t16.ExtractBytes()
	
	if len(bytes) != 4 {
		t.Fatalf("Expected 4 bytes, got %d", len(bytes))
	}
	
	// Verify generic values (Little Endian)
	v1 := binary.LittleEndian.Uint16(bytes[0:2])
	v2 := binary.LittleEndian.Uint16(bytes[2:4])
	
	// 1.0 in FP16 = 0x3c00
	if v1 != 0x3c00 {
		t.Errorf("Expected 0x3c00 for 1.0, got 0x%x", v1)
	}
	
	// -2.0 in FP16 = 0xc000
	if v2 != 0xc000 {
		t.Errorf("Expected 0xc000 for -2.0, got 0x%x", v2)
	}
	
	// Cleanup
	backend.PutTensor(t16)
	backend.PutTensor(tensor)
}

func TestMetal_Cast_FP16_to_FP16_NoOp(t *testing.T) {
	// If backing is already FP16, Cast(Float16) should clone or return efficient copy
	backend := NewMetalBackendFP16()
	data := []float32{1.0, -2.0}
	tensor := backend.NewTensor(1, 2, data)
	
	t16 := tensor.Cast(Float16)
	
	// It should be a new tensor (copy) given current implementation logic
	if t16 == tensor {
		t.Log("Warning: Cast returned same tensor (pointer equal)")
	}
	
	bytes := t16.ExtractBytes()
	v1 := binary.LittleEndian.Uint16(bytes[0:2])
	if v1 != 0x3c00 {
		t.Errorf("Expected 0x3c00, got 0x%x", v1)
	}
}
