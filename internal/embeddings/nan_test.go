package embeddings

import (
	"fmt"
	"testing"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/device"
	"math"
)

func TestNaNGuard(t *testing.T) {
	// Setup
	config := model.DefaultBertTinyConfig()
	backend := device.NewCPUBackend() 
	bert := model.NewBertModelWithBackend(config, backend)
	


	// Manually construct a tensor with NaN
	nanTensor := backend.NewTensor(1, 128, nil)
	// Inject NaN
	cpuT, _ := nanTensor.(*device.CPUTensor)
	cpuT.Set(0, 0, float32(math.NaN()))
	
	hasNaN, _ := nanTensor.HasNaN()
	if !hasNaN {
		t.Errorf("Backend failed to detect NaN")
	}
	
	// Test Embedder Integration via a mocked forward pass? 
	// Difficult without mocking model.ForwardBatch.
	// But we can check processOutput directly.
	
	outCh := make(chan StreamResult, 1)
	indices := []int{0}
	
	// Case 1: Validation Error passed
	processOutput(bert, nanTensor, 128, indices, "fp32", outCh, nil, nil, fmt.Errorf("NaN Detected"))
	
	res := <-outCh
	if res.Err == nil {
		t.Errorf("Expected error in StreamResult, got nil")
	}
	if res.Err.Error() != "NaN Detected" {
		t.Errorf("Expected 'NaN Detected', got %v", res.Err)
	}

	// Case 2: No Error
	normalTensor := backend.NewTensor(1, 128, nil)
	processOutput(bert, normalTensor, 128, indices, "fp32", outCh, nil, nil, nil)
	res = <-outCh
	if res.Err != nil {
		t.Errorf("Unexpected error: %v", res.Err)
	}
}
