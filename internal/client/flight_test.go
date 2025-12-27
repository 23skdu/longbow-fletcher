package client

import (
	"context"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockFlightServer struct {
	flight.BaseFlightServer
	recordsReceived []arrow.Record
}

func (s *mockFlightServer) DoPut(server flight.FlightService_DoPutServer) error {
	for {
		batch, err := server.Recv()
		if err != nil {
			return nil
		}
		
		record, err := flight.NewRecordReader(server)
		if err != nil {
			return err
		}
		
		for record.Next() {
			rec := record.Record()
			rec.Retain()
			s.recordsReceived = append(s.recordsReceived, rec)
		}
		
		_ = batch // We can check the descriptor here if needed
	}
}

func TestFlightClient_DoPut(t *testing.T) {
	// Start a mock flight server
	mockServer := &mockFlightServer{}
	server := flight.NewServerWithMiddleware(nil)
	server.RegisterFlightService(mockServer)
	
	err := server.Init("localhost:0")
	require.NoError(t, err)
	addr := server.Addr().String()
	
	go func() {
		_ = server.Serve()
	}()
	defer server.Shutdown()

	client, err := NewFlightClient(addr)
	require.NoError(t, err)
	defer client.Close()

	pool := memory.NewGoAllocator()
	schema := arrow.NewSchema(
		[]arrow.Field{{Name: "f1", Type: arrow.PrimitiveTypes.Float32}},
		nil,
	)
	b := array.NewFloat32Builder(pool)
	defer b.Release()
	b.AppendValues([]float32{1.0, 2.0}, nil)
	a := b.NewArray()
	defer a.Release()
	
	rb := array.NewRecordBatch(schema, []arrow.Array{a}, 2)
	defer rb.Release()

	err = client.DoPut(context.Background(), "test-dataset", rb)
	// We might need a small sleep or retry if server isn't ready
	assert.NoError(t, err)
}
