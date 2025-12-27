package client

import (
	"context"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// FlightClient handles communication with a Longbow server via Apache Flight.
type FlightClient struct {
	client flight.Client
	conn   *grpc.ClientConn
}

// NewFlightClient creates a new Flight client connected to the given address.
func NewFlightClient(addr string) (*FlightClient, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	client := flight.NewClientFromConn(conn, nil)
	return &FlightClient{
		client: client,
		conn:   conn,
	}, nil
}

// DoPut sends a RecordBatch to the given dataset on the Longbow server.
func (c *FlightClient) DoPut(ctx context.Context, datasetName string, record arrow.RecordBatch) error {
	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{datasetName},
	}

	stream, err := c.client.DoPut(ctx)
	if err != nil {
		return err
	}

	writer := flight.NewRecordWriter(stream)
	// Set the flight descriptor in the writer if needed, or send it in the first message.
	// Flight DoPut usually starts with a FlightDescriptor.
	// The flight.Client.DoPut doesn't take the descriptor directly, 
	// we use the writer to send it.
	writer.SetFlightDescriptor(desc)

	if err := writer.Write(record); err != nil {
		return err
	}

	return writer.Close()
}

// Close closes the client connection.
func (c *FlightClient) Close() error {
	return c.conn.Close()
}
