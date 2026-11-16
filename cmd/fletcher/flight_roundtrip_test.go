//go:build integration

package main

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/client"
	"github.com/23skdu/longbow-fletcher/internal/embeddings"
)

func TestFlightRoundTripIntegration(t *testing.T) {
	weights := "../../bert_tiny.safetensors"
	if _, err := os.Stat(weights); err != nil {
		t.Skip("weights not available")
	}

	vocab, err := os.CreateTemp("", "vocab-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(vocab.Name())
	for _, w := range []string{"[CLS]", "[SEP]", "[UNK]", "test", "hello", "world"} {
		fmt.Fprintln(vocab, w)
	}
	vocab.Close()

	emb, err := embeddings.NewEmbedder(vocab.Name(), weights, false, "bert-tiny", "fp32")
	if err != nil {
		t.Fatal(err)
	}

	listenAddr := "127.0.0.1:18099"
	go StartFlightServer(listenAddr, emb, nil)

	var ec *client.EmbeddingClient
	for i := 0; i < 50; i++ {
		c, err := client.NewEmbeddingClient(listenAddr)
		if err == nil {
			ec = c
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if ec == nil {
		t.Fatal("could not connect")
	}
	defer ec.Close()

	vecs, err := ec.Embed(context.Background(), []string{"hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 1 || len(vecs[0]) == 0 {
		t.Fatalf("unexpected result: %d", len(vecs))
	}
}
