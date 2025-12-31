package embeddings

import (
	"context"
	"os"
	"testing"


	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/tokenizer"
	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/cache"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func getMetricValue(m prometheus.Metric) float64 {
	var metric dto.Metric
	m.Write(&metric)
	if metric.Counter != nil {
		return *metric.Counter.Value
	}
	if metric.Gauge != nil {
		return *metric.Gauge.Value
	}
	return 0
}

func TestEmbedder_Caching(t *testing.T) {
	// Setup
	vocabPath := createTempVocab(t) // reuse helper from embeddings_test.go
	defer func() { _ = os.Remove(vocabPath) }()
	
	tok, _ := tokenizer.NewWordPieceTokenizer(vocabPath)
	config := model.DefaultBertTinyConfig()
	backend := device.NewCPUBackend()
	bert := model.NewBertModelWithBackend(config, backend)
	
	c := cache.NewMapCache()
	
	e := &Embedder{
		models:            []*model.BertModel{bert},
		tokenizer:         tok,
		internalBatchSize: 32,
		maxBatchTokens:    1024,
		cache:             c,
		gpuMetrics:        make([]GPUMetrics, 1),
	}
	
	// Dataset ID
	ctx := WithDatasetID(context.Background(), "ds-123")
	
	// 1. First Pass: "hello"
	// Should be Miss
	startHits := getMetricValue(cacheHits)
	startMisses := getMetricValue(cacheMisses)
	
	res1 := e.ProxyEmbedBatch(ctx, []string{"hello"})
	if len(res1) == 0 {
		t.Fatal("Empty result 1")
	}
	
	if hit, miss := getMetricValue(cacheHits), getMetricValue(cacheMisses); miss-startMisses != 1 {
		t.Errorf("Expected 1 miss, got %v (hits=%v)", miss-startMisses, hit-startHits)
	}
	
	// 2. Second Pass: "hello", "world"
	// "hello" should be Hit
	// "world" should be Miss
	
	res2 := e.ProxyEmbedBatch(ctx, []string{"hello", "world"})
	
	if len(res2) != 2*config.HiddenSize {
		t.Fatal("Result size mismatch")
	}
	
	if hit, miss := getMetricValue(cacheHits), getMetricValue(cacheMisses); hit-startHits != 1 {
		t.Errorf("Expected 1 total hit (previous 0 + 1), got %v", hit-startHits)
	} else if miss-startMisses != 2 {
		t.Errorf("Expected 2 total misses (previous 1 + 1), got %v", miss-startMisses)
	}
	
	// Verify Values match
	// res1[0..D] should equal res2[0..D]
	for i := 0; i < config.HiddenSize; i++ {
		if res1[i] != res2[i] {
			t.Errorf("Cache consistency error at %d: %f != %f", i, res1[i], res2[i])
		}
	}
	
	// 3. Different Dataset ID
	// "hello" should be Miss (different key)
	ctx2 := WithDatasetID(context.Background(), "ds-456")
	e.ProxyEmbedBatch(ctx2, []string{"hello"})
	
	if _, miss := getMetricValue(cacheHits), getMetricValue(cacheMisses); miss-startMisses != 3 {
		t.Errorf("Expected 3 total misses (previous 2 + 1), got %v", miss-startMisses)
	}
}
