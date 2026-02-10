package client

import (
	"context"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type EmbeddingClient struct {
	client flight.Client
	alloc  memory.Allocator
	schema *arrow.Schema
}

func NewEmbeddingClient(addr string) (*EmbeddingClient, error) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	}
	
	client, err := flight.NewClientWithMiddleware(addr, nil, nil, opts...)
	if err != nil {
		return nil, err
	}
	
	return &EmbeddingClient{
		client: client,
		alloc:  memory.NewGoAllocator(),
		schema: arrow.NewSchema(
			[]arrow.Field{
				{Name: "row_id", Type: arrow.PrimitiveTypes.Uint64},
				{Name: "text", Type: arrow.BinaryTypes.String},
			},
			nil,
		),
	}, nil
}

func (c *EmbeddingClient) Close() error {
	return c.client.Close()
}

func (c *EmbeddingClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	// Simple implementation of DoExchange client
	stream, err := c.client.DoExchange(ctx)
	if err != nil {
		return nil, err
	}
	
	// Write Request
	writer := flight.NewRecordWriter(stream, ipc.WithSchema(c.schema))
	
	builder := array.NewRecordBuilder(c.alloc, c.schema)
	defer builder.Release()
	
	idBuilder := builder.Field(0).(*array.Uint64Builder)
	textBuilder := builder.Field(1).(*array.StringBuilder)
	
	for i, t := range texts {
		idBuilder.Append(uint64(i))
		textBuilder.Append(t)
	}
	
	rec := builder.NewRecordBatch()
	defer rec.Release()
	
	if err := writer.Write(rec); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	
	// Read Response
	reader, err := flight.NewRecordReader(stream, ipc.WithAllocator(c.alloc))
	if err != nil {
		return nil, err
	}
	defer reader.Release()
	
	results := make([][]float32, len(texts))
	
	for reader.Next() {
		outRec := reader.Record()
		rowCount := int(outRec.NumRows())
		if rowCount == 0 { continue }
		
		ids := outRec.Column(0).(*array.Uint64)
		embs := outRec.Column(1).(*array.FixedSizeList)
		values := embs.ListValues().(*array.Float32)
		listSize := int(embs.DataType().(*arrow.FixedSizeListType).Len())
		
		for i := 0; i < rowCount; i++ {
			id := ids.Value(i)
			if int(id) < len(results) {
				start := i * listSize
				vec := make([]float32, listSize)
				// Copy
				copy(vec, values.Float32Values()[start:start+listSize])
				results[id] = vec
			}
		}
	}
	
	return results, reader.Err()
}
