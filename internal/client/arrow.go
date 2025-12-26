package client

import (
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// RecordBatchBuilder creates Arrow RecordBatches from embeddings.
type RecordBatchBuilder struct {
	mem memory.Allocator
}

// NewRecordBatchBuilder creates a new builder.
func NewRecordBatchBuilder(mem memory.Allocator) *RecordBatchBuilder {
	return &RecordBatchBuilder{mem: mem}
}

// BuildRecordBatch converts a slice of embeddings into a RecordBatch.
// Each embedding is expected to be a []float32 or []float64.
// For Longbow, we typically want a schema with at least a vector column.
func (b *RecordBatchBuilder) BuildRecordBatch(embeddings [][]float32) (arrow.Record, error) {
	if len(embeddings) == 0 {
		return nil, nil
	}

	numRows := len(embeddings)

	// Define Schema
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "vector", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	// Build Vector Column
	listBuilder := array.NewListBuilder(b.mem, arrow.PrimitiveTypes.Float32)
	defer listBuilder.Release()
	
	valueBuilder := listBuilder.ValueBuilder().(*array.Float32Builder)

	for _, emb := range embeddings {
		listBuilder.Append(true)
		valueBuilder.AppendValues(emb, nil)
	}

	cols := []arrow.Array{listBuilder.NewArray()}
	defer cols[0].Release()

	return array.NewRecord(schema, cols, int64(numRows)), nil
}
