package model

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

// BufferPool provides pooled matrices for intermediate computations.
// This dramatically reduces allocations during embedding generation.
type BufferPool struct {
	// Pools for different sized matrices
	seqHidden sync.Pool // seqLen x hiddenSize
	seqSeq    sync.Pool // seqLen x seqLen
	seqInter  sync.Pool // seqLen x intermediateSize
}

// Global buffer pool
var Pool = &BufferPool{}

// GetSeqHidden gets a seqLen x hiddenSize matrix from the pool.
func (p *BufferPool) GetSeqHidden(rows, cols int) *mat.Dense {
	if v := p.seqHidden.Get(); v != nil {
		m := v.(*mat.Dense)
		r, c := m.Dims()
		if r >= rows && c == cols {
			// Zero only the rows we need
			raw := m.RawMatrix().Data[:rows*cols]
			for i := range raw {
				raw[i] = 0
			}
			return mat.NewDense(rows, cols, raw)
		}
	}
	return mat.NewDense(rows, cols, nil)
}

// PutSeqHidden returns a matrix to the pool.
func (p *BufferPool) PutSeqHidden(m *mat.Dense) {
	if m != nil {
		p.seqHidden.Put(m)
	}
}

// GetSeqSeq gets a seqLen x seqLen matrix from the pool.
func (p *BufferPool) GetSeqSeq(size int) *mat.Dense {
	if v := p.seqSeq.Get(); v != nil {
		m := v.(*mat.Dense)
		r, c := m.Dims()
		if r >= size && c >= size {
			raw := m.RawMatrix().Data[:size*size]
			for i := range raw {
				raw[i] = 0
			}
			return mat.NewDense(size, size, raw)
		}
	}
	return mat.NewDense(size, size, nil)
}

// PutSeqSeq returns a matrix to the pool.
func (p *BufferPool) PutSeqSeq(m *mat.Dense) {
	if m != nil {
		p.seqSeq.Put(m)
	}
}

// GetSeqInter gets a seqLen x intermediateSize matrix from the pool.
func (p *BufferPool) GetSeqInter(rows, cols int) *mat.Dense {
	if v := p.seqInter.Get(); v != nil {
		m := v.(*mat.Dense)
		r, c := m.Dims()
		if r >= rows && c == cols {
			raw := m.RawMatrix().Data[:rows*cols]
			for i := range raw {
				raw[i] = 0
			}
			return mat.NewDense(rows, cols, raw)
		}
	}
	return mat.NewDense(rows, cols, nil)
}

// PutSeqInter returns a matrix to the pool.
func (p *BufferPool) PutSeqInter(m *mat.Dense) {
	if m != nil {
		p.seqInter.Put(m)
	}
}
