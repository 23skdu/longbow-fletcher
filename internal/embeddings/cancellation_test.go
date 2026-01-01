package embeddings

import (
	"context"
	"testing"
	"time"
	"fmt"
	"os"
	
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
)

func TestEmbedder_Cancellation(t *testing.T) {
	// Setup
	vocabPath := createTempVocab(t)
	defer func() { _ = os.Remove(vocabPath) }()
	tok, _ := tokenizer.NewWordPieceTokenizer(vocabPath)
	config := model.DefaultBertTinyConfig()
	backend := device.NewCPUBackend() 
	bert := model.NewBertModelWithBackend(config, backend)
	
	e := &Embedder{
		models:            []*model.BertModel{bert},
		tokenizer:         tok,
		internalBatchSize: 1, // Force small batches to ensure multiple iterations
		maxBatchTokens:    512,
		gpuMetrics:        make([]GPUMetrics, 1),
	}
	
	// Create a large input to force multiple batches
	in := make([]string, 100)
	for i := range in {
		in[i] = fmt.Sprintf("text %d", i)
	}
	
	// Context with timeout too short to finish all
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()
	
	out := e.EmbedBatch(ctx, in)

	// Consume results
	count := 0
	for range out {
		count++
	}
	
	if count == 100 {
		t.Errorf("Expected cancellation to stop processing, but got all %d results", count)
	}
	t.Logf("Processed %d/%d before cancellation", count, 100)
}
