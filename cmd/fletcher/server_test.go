package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/embeddings"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/fxamacker/cbor/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type mockFlightClient struct {
	mock.Mock
}

func (m *mockFlightClient) DoPut(ctx context.Context, datasetName string, record arrow.RecordBatch) error {
	args := m.Called(ctx, datasetName, record)
	return args.Error(0)
}

func (m *mockFlightClient) Close() error {
	return nil
}

func TestServer_Full(t *testing.T) {
	// 1. Create dummy vocab
	vocabFile := "test_vocab.txt"
	os.WriteFile(vocabFile, []byte("[CLS]\n[SEP]\n[UNK]\ntest\n"), 0644)
	defer os.Remove(vocabFile)

	// 2. Create minimal embedder (CPU, bert-tiny)
	// We skip weight loading by passing empty path
	emb, err := embeddings.NewEmbedder(vocabFile, "", false, "bert-tiny", "fp32")
	assert.NoError(t, err)

	// 43: srv := NewServer(emb, mfc, "test-dataset", 100, 0) // Unlimited VRAM for full test
	// 43: srv := NewServer(emb, mfc, "test-dataset", 100, 0) // Unlimited VRAM for full test
	mfc := &mockFlightClient{}
	srv := NewServer(emb, mfc, "test-dataset", 100, 0, "fp32")

	t.Run("HandleEncode with Forwarding", func(t *testing.T) {
		texts := []string{"test", "test"}
		data, _ := cbor.Marshal(texts)
		req, _ := http.NewRequest("POST", "/encode", bytes.NewReader(data))
		rr := httptest.NewRecorder()

		// Expect DoPut to be called
		mfc.On("DoPut", mock.Anything, "test-dataset", mock.Anything).Return(nil)

		http.HandlerFunc(srv.handleEncode).ServeHTTP(rr, req)

		assert.Equal(t, http.StatusOK, rr.Code)
		mfc.AssertExpectations(t)
	})

	t.Run("Health Check", func(t *testing.T) {
		req, _ := http.NewRequest("GET", "/health", nil)
		rr := httptest.NewRecorder()
		
		srv.handleHealth(rr, req)
		assert.Equal(t, http.StatusOK, rr.Code)
		assert.Equal(t, "OK", rr.Body.String())
	})
}

type mockEmbedder struct {
	mock.Mock
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string) <-chan embeddings.StreamResult {
	args := m.Called(ctx, texts)
	return args.Get(0).(<-chan embeddings.StreamResult)
}
func (m *mockEmbedder) ProxyEmbedBatch(ctx context.Context, texts []string) []float32 {
	args := m.Called(ctx, texts)
	return args.Get(0).([]float32)
}
func (m *mockEmbedder) GetVRAMUsage() (int64, int64) {
	args := m.Called()
	return args.Get(0).(int64), args.Get(1).(int64)
}
func (m *mockEmbedder) EstimateVRAM(numSequences int, totalBytes int) int64 {
	args := m.Called(numSequences, totalBytes)
	return args.Get(0).(int64)
}

func TestServer_VRAMAdmission(t *testing.T) {
	emb := &mockEmbedder{}
	mfc := &mockFlightClient{}
	
	// Create server with 100 bytes VRAM limit
	srv := NewServer(emb, mfc, "test-dataset", 10, 100, "fp32")
	
	t.Run("Reject Request Exceeding VRAM", func(t *testing.T) {
		// Mock estimation to return 200 bytes (Over limit)
		emb.On("EstimateVRAM", 1, 4).Return(int64(200)).Once()
		
		texts := []string{"test"}
		data, _ := cbor.Marshal(texts)
		req, _ := http.NewRequest("POST", "/encode", bytes.NewReader(data))
		// Use a timeout context to prevent deadlock if Acquire blocks forever
		ctx, cancel := context.WithTimeout(req.Context(), 100*time.Millisecond)
		defer cancel()
		req = req.WithContext(ctx)
		
		rr := httptest.NewRecorder()
		
		http.HandlerFunc(srv.handleEncode).ServeHTTP(rr, req)
		
		// Should fail with 503
		assert.Equal(t, http.StatusServiceUnavailable, rr.Code)
		emb.AssertExpectations(t)
	})

	t.Run("Accept Request Within VRAM", func(t *testing.T) {
		// Mock estimation to return 50 bytes (Under limit)
		emb.On("EstimateVRAM", 1, 4).Return(int64(50)).Once()
		
		// Mock EmbedBatch to return empty channel
		ch := make(chan embeddings.StreamResult)
		close(ch)
		emb.On("EmbedBatch", mock.Anything, []string{"test"}).Return((<-chan embeddings.StreamResult)(ch)).Once()
		
		texts := []string{"test"}
		data, _ := cbor.Marshal(texts)
		req, _ := http.NewRequest("POST", "/encode", bytes.NewReader(data))
		rr := httptest.NewRecorder()
		
		http.HandlerFunc(srv.handleEncode).ServeHTTP(rr, req)
		
		// Should succeed
		assert.Equal(t, http.StatusOK, rr.Code)
		emb.AssertExpectations(t)
	})
}

func TestServer_FP16Transport(t *testing.T) {
	emb := &mockEmbedder{}
	mfc := &mockFlightClient{}
	
	// Create server with FP16 transport
	srv := NewServer(emb, mfc, "test-dataset", 10, 0, "fp16")
	
	t.Run("Verify FP16 Conversion", func(t *testing.T) {
		// Mock EmbedBatch to return a chunk with known values
		ch := make(chan embeddings.StreamResult, 1)
		// 1 text, 2 dimensions
		vectors := []float32{1.0, -2.0}
		ch <- embeddings.StreamResult{
			Vectors: vectors,
			Count: 1,
			Offset: 0,
			Err: nil,
		}
		close(ch)
		
		emb.On("EmbedBatch", mock.Anything, []string{"test"}).Return((<-chan embeddings.StreamResult)(ch)).Once()
		emb.On("EstimateVRAM", mock.Anything, mock.Anything).Return(int64(0)).Maybe() // if called
		
		// Validate DoPut Arguments
		mfc.On("DoPut", mock.Anything, "test-dataset", mock.MatchedBy(func(rb arrow.RecordBatch) bool {
			// Check Schema
			schema := rb.Schema()
			if !assert.Equal(t, 2, len(schema.Fields())) { return false }
			if !assert.Equal(t, "embedding", schema.Field(1).Name) { return false }
			
			// Check Type is FixedSizeList<Float16>
			fsl, ok := schema.Field(1).Type.(*arrow.FixedSizeListType)
			if !assert.True(t, ok, "Expected FixedSizeList") { return false }
			if !assert.Equal(t, arrow.FixedWidthTypes.Float16, fsl.Elem()) { return false }
			
			return true
		})).Return(nil).Once()
		
		texts := []string{"test"}
		data, _ := cbor.Marshal(texts)
		req, _ := http.NewRequest("POST", "/encode", bytes.NewReader(data))
		rr := httptest.NewRecorder()
		
		http.HandlerFunc(srv.handleEncode).ServeHTTP(rr, req)
		
		assert.Equal(t, http.StatusOK, rr.Code)
		mfc.AssertExpectations(t)
	})
}
