package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/23skdu/longbow-fletcher/internal/client"
	"github.com/23skdu/longbow-fletcher/internal/embeddings"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/fxamacker/cbor/v2"
)

type Server struct {
	embedder     *embeddings.Embedder
	flightClient *client.FlightClient
	datasetName  string
}

func startServer(addr string, embedder *embeddings.Embedder, fc *client.FlightClient, dataset string) {
	srv := &Server{
		embedder:     embedder,
		flightClient: fc,
		datasetName:  dataset,
	}

	http.HandleFunc("/encode", srv.handleEncode)
	
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	log.Printf("Starting Fletcher Server on %s", addr)
	if fc != nil {
		log.Printf("Forwarding to Longbow at specified server address")
	}

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func (s *Server) handleEncode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var texts []string
	decoder := cbor.NewDecoder(r.Body)
	if err := decoder.Decode(&texts); err != nil {
		http.Error(w, fmt.Sprintf("Bad Request (CBOR decode): %v", err), http.StatusBadRequest)
		return
	}

	if len(texts) == 0 {
		w.WriteHeader(http.StatusOK)
		return
	}

	// 2. Embed -> [][]float32
	batch := s.embedder.EmbedBatch(texts)

	// 3. Forward to Longbow
	if s.flightClient != nil {
		if err := s.forwardToLongbow(r.Context(), texts, batch); err != nil {
			log.Printf("Error forwarding to Longbow: %v", err)
			http.Error(w, "Error forwarding to Longbow", http.StatusInternalServerError)
			return
		}
	}

	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}

func (s *Server) forwardToLongbow(ctx context.Context, texts []string, batch [][]float32) error {
	pool := memory.NewGoAllocator()
	curBatchSize := len(texts)
	
	// Text Column
	tb := array.NewStringBuilder(pool)
	defer tb.Release()
	tb.AppendValues(texts, nil)
	textArr := tb.NewArray()
	defer textArr.Release()

	// Embedding Column
	if len(batch) == 0 {
		return nil
	}
	cols := len(batch[0])
	
	// Flatten
	flatData := make([]float32, 0, curBatchSize*cols)
	for _, vec := range batch {
		if len(vec) != cols {
			// Should not happen, but safe to check
			continue 
		}
		flatData = append(flatData, vec...)
	}

	fb := array.NewFloat32Builder(pool)
	defer fb.Release()
	fb.AppendValues(flatData, nil)
	values := fb.NewArray()
	defer values.Release()
	
	fslType := arrow.FixedSizeListOf(int32(cols), arrow.PrimitiveTypes.Float32)
	
	fslData := array.NewData(
		fslType,
		curBatchSize,
		[]*memory.Buffer{nil}, 
		[]arrow.ArrayData{values.Data()},
		0,
		0,
	)
	defer fslData.Release()
	embeddingArr := array.NewFixedSizeListData(fslData)
	defer embeddingArr.Release()

	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "text", Type: arrow.BinaryTypes.String},
			{Name: "embedding", Type: fslType},
		},
		nil,
	)
	
	rb := array.NewRecordBatch(schema, []arrow.Array{textArr, embeddingArr}, int64(curBatchSize))
	defer rb.Release()
	
	return s.flightClient.DoPut(ctx, s.datasetName, rb)
}
