package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

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

	mfc := &mockFlightClient{}

	srv := &Server{
		embedder:     emb,
		flightClient: mfc,
		datasetName:  "test-dataset",
	}

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
