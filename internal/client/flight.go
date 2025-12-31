package client

import (
	"context"
	"sync"
	"time"

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

// AsyncStream handles background streaming of records.
type AsyncStream struct {
	ch      chan arrow.RecordBatch
	client  *FlightClient
	dataset string
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

// StartStream begins a background thread that streams records to the server.
// It maintains a persistent connection, reconnecting on failure.
func (c *FlightClient) StartStream(ctx context.Context, datasetName string) *AsyncStream {
	ctx, cancel := context.WithCancel(ctx)
	s := &AsyncStream{
		ch:      make(chan arrow.RecordBatch, 256), // Buffer for burst absorption
		client:  c,
		dataset: datasetName,
		ctx:     ctx,
		cancel:  cancel,
	}
	
	s.wg.Add(1)
	go s.run()
	return s
}

// Send queues a record for streaming. It retains the record, so the caller
// can release it immediately after calling Send.
func (s *AsyncStream) Send(rec arrow.RecordBatch) {
	rec.Retain()
	select {
	case s.ch <- rec:
	case <-s.ctx.Done():
		rec.Release() // Drain
	}
}

// Close stops the stream and waits for the background thread to finish.
func (s *AsyncStream) Close() {
	s.cancel()
	s.wg.Wait()
}

func (s *AsyncStream) run() {
	defer s.wg.Done()
	
	var stream flight.FlightService_DoPutClient
	var writer *flight.Writer
	var err error
	
	// Helper to close current stream
	reset := func() {
		if writer != nil {
			writer.Close()
			writer = nil
		}
		if stream != nil {
			stream.CloseSend() // Best effort
			stream = nil
		}
	}
	defer reset() // Cleanup on exit

	for {
		// 1. Establish Stream if needed
		if writer == nil {
			desc := &flight.FlightDescriptor{
				Type: flight.DescriptorPATH,
				Path: []string{s.dataset},
			}
			
			// We use a separate context for the stream I/O so we can cancel it
			// independently or it ends when s.ctx ends.
			stream, err = s.client.client.DoPut(s.ctx)
			if err != nil {
				// Backoff and retry
				select {
				case <-s.ctx.Done():
					return
				case <-time.After(time.Second):
					continue
				}
			}
			
			writer = flight.NewRecordWriter(stream)
			writer.SetFlightDescriptor(desc)
		}
		
		// 2. Read from channel
		select {
		case <-s.ctx.Done():
			return
		case rec := <-s.ch:
			// 3. Write
			if err := writer.Write(rec); err != nil {
				// Write failed. Requeue? 
				// For now, we drop and reconnect to avoid head-of-line blocking 
				// or complex buffer management.
				// Ideally, we'd log this.
				rec.Release()
				reset()
				// Loop will reconnect
			} else {
				rec.Release()
			}
		}
	}
}


// Close closes the client connection.
func (c *FlightClient) Close() error {
	return c.conn.Close()
}
