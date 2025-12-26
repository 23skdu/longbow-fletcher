package model

import (
	"testing"
)

func TestBufferPool(t *testing.T) {
	pool := &BufferPool{}

	// Test SeqHidden pool
	rows, cols := 5, 10
	m1 := pool.GetSeqHidden(rows, cols)
	
	r, c := m1.Dims()
	if r != rows || c != cols {
		t.Errorf("GetSeqHidden dimensions = %dx%d, want %dx%d", r, c, rows, cols)
	}

	// Put back
	pool.PutSeqHidden(m1)

	// Get again, should potentially be the same object (though sync.Pool doesn't guarantee it)
	// mostly checking it doesn't panic and returns correct dims
	m2 := pool.GetSeqHidden(rows, cols)
	r2, c2 := m2.Dims()
	if r2 != rows || c2 != cols {
		t.Errorf("GetSeqHidden (2nd) dimensions = %dx%d, want %dx%d", r2, c2, rows, cols)
	}
}

func TestBufferPool_Resize(t *testing.T) {
	// Test that requesting different size returns correct size
	// sync.Pool might return a recycled larger matrix if not careful, 
	// but our implementation should handle resizing or new allocation check?
	// Actually our implementation in pool.go uses `mat.NewDense` or reused one.
	// Let's check pool.go implementation details (I recall it has pools for specific logical "slots" but not exact sizes?)
	// Looking at previous valid code, it had specific pools seqHidden, seqSeq, seqInter.
	// We assume fixed max sizes or dynamic reset?
	// Let's just verify it returns correct dimensions.
	
	pool := &BufferPool{}
	m := pool.GetSeqHidden(10, 20)
	if r, c := m.Dims(); r != 10 || c != 20 {
		t.Errorf("GetSeqHidden(10, 20) = %dx%d", r, c)
	}
	pool.PutSeqHidden(m)
	
	// Different size (though in BERT fixed workflow sizes don't change much)
	m2 := pool.GetSeqHidden(5, 5)
	if r, c := m2.Dims(); r != 5 || c != 5 {
		t.Errorf("GetSeqHidden(5, 5) = %dx%d", r, c)
	}
}
