package main

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/rs/zerolog/log"
)

type FletcherFlightServer struct {
	flight.BaseFlightServer
	embedder EmbedderInterface
	alloc    memory.Allocator
}

func NewFletcherFlightServer(embedder EmbedderInterface) *FletcherFlightServer {
	return &FletcherFlightServer{
		embedder: embedder,
		alloc:    memory.NewGoAllocator(),
	}
}

func (s *FletcherFlightServer) DoExchange(stream flight.FlightService_DoExchangeServer) error {
	return fmt.Errorf("DoExchange not implemented")
}

func (s *FletcherFlightServer) DoPut(stream flight.FlightService_DoPutServer) error {
	reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(s.alloc))
	if err != nil {
		return err
	}
	defer reader.Release()

	for reader.Next() {
		rec := reader.Record()
		log.Info().Int64("rows", rec.NumRows()).Msg("DoPut received batch")
		// TODO: Implement embedding logic here
	}
	return reader.Err()
}

func StartFlightServer(addr string, embedder EmbedderInterface) {
	// Create the generic Flight Server which manages the GRPC lifecycle
	server := flight.NewFlightServer()
	
	// Register our custom service implementation
	server.RegisterFlightService(NewFletcherFlightServer(embedder))
	
	// Init handles the listener creation internally
	if err := server.Init(addr); err != nil {
		log.Fatal().Err(err).Msg("Failed to init Flight server")
	}

	log.Info().Str("addr", addr).Msg("Starting Fletcher Flight Server")
	if err := server.Serve(); err != nil {
		log.Fatal().Err(err).Msg("Flight server failed")
	}
}
