//go:build ignore

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/client"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	addr := "localhost:9090"
	if len(os.Args) > 1 {
		addr = os.Args[1]
	}

	log.Info().Str("addr", addr).Msg("Connecting to Fletcher Flight Server")

	// Retry connection loop
	var c *client.EmbeddingClient
	var err error

	for i := 0; i < 10; i++ {
		c, err = client.NewEmbeddingClient(addr)
		if err == nil {
			break
		}
		log.Warn().Err(err).Msg("Connection failed, retrying...")
		time.Sleep(1 * time.Second)
	}

	if err != nil {
		log.Fatal().Err(err).Msg("Failed to connect after retries")
	}
	defer c.Close()

	texts := []string{
		"Hello world",
		"Apache Arrow Flight is fast",
		"Embedding vectors are useful",
	}

	log.Info().Int("count", len(texts)).Msg("Sending texts")

	start := time.Now()
	embeddings, err := c.Embed(context.Background(), texts)
	if err != nil {
		log.Fatal().Err(err).Msg("Embed failed")
	}
	elapsed := time.Since(start)

	log.Info().Dur("elapsed", elapsed).Msg("Received embeddings")

	if len(embeddings) != len(texts) {
		log.Fatal().Int("expected", len(texts)).Int("got", len(embeddings)).Msg("Count mismatch")
	}

	for i, vec := range embeddings {
		if len(vec) != 128 {
			log.Fatal().Int("index", i).Int("dim", len(vec)).Msg("Dimension mismatch (expected 128)")
		}
		log.Info().Int("index", i).Int("dim", len(vec)).Msg("Vector valid")
	}

	fmt.Println("VERIFICATION PASSED")
}
