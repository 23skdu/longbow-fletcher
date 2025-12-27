package client

import (
	"testing"

	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/assert"
)

func TestBuildRecordBatch(t *testing.T) {
	pool := memory.NewGoAllocator()
	builder := NewRecordBatchBuilder(pool)

	t.Run("Empty input", func(t *testing.T) {
		rb, err := builder.BuildRecordBatch(nil)
		assert.NoError(t, err)
		assert.Nil(t, rb)
	})

	t.Run("Valid input", func(t *testing.T) {
		embeddings := [][]float32{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
		}

		rb, err := builder.BuildRecordBatch(embeddings)
		assert.NoError(t, err)
		assert.NotNil(t, rb)
		defer rb.Release()

		assert.Equal(t, int64(2), rb.NumRows())
		assert.Equal(t, int64(1), rb.NumCols())
		assert.Equal(t, "vector", rb.ColumnName(0))

		// Check data
		listArr := rb.Column(0).(*array.List)
		assert.Equal(t, 2, listArr.Len())

		offsets := listArr.Offsets()
		assert.Equal(t, []int32{0, 3, 6}, offsets)

		values := listArr.ListValues().(*array.Float32)
		assert.Equal(t, 6, values.Len())
		assert.Equal(t, float32(1.0), values.Value(0))
		assert.Equal(t, float32(6.0), values.Value(5))
	})
}
