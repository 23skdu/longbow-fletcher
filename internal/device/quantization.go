package device

import (
	"math"
	"runtime"
	"sync"
)

type QuantizationType int

const (
	QuantNone QuantizationType = iota
	QuantInt8
	QuantInt4
)

type QuantizedTensor struct {
	data   []byte
	scales []float32
	zeros  []float32
	rows   int
	cols   int
	qtype  QuantizationType
}

func (t *QuantizedTensor) Dims() (int, int) {
	return t.rows, t.cols
}

func (t *QuantizedTensor) ToFloat32() []float32 {
	out := make([]float32, t.rows*t.cols)
	switch t.qtype {
	case QuantInt8:
		t.dequantizeInt8(out)
	case QuantInt4:
		t.dequantizeInt4(out)
	}
	return out
}

func (t *QuantizedTensor) dequantizeInt8(out []float32) {
	numBlocks := len(t.scales)
	blockSize := (t.rows * t.cols) / numBlocks

	parallelize := func(fn func(start, end int)) {
		numWorkers := runtime.NumCPU()
		chunkSize := (numBlocks + numWorkers - 1) / numWorkers
		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			start := w * chunkSize
			end := start + chunkSize
			if end > numBlocks {
				end = numBlocks
			}
			if start >= numBlocks {
				break
			}
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				fn(s, e)
			}(start, end)
		}
		wg.Wait()
	}

	parallelize(func(start, end int) {
		for b := start; b < end; b++ {
			scale := t.scales[b]
			zero := t.zeros[b]
			blockStart := b * blockSize
			for i := 0; i < blockSize; i++ {
				val := int8(t.data[blockStart+i])
				out[blockStart+i] = float32(val)*scale + zero
			}
		}
	})
}

func (t *QuantizedTensor) dequantizeInt4(out []float32) {
	numBlocks := len(t.scales)
	blockSize := (t.rows * t.cols) / numBlocks

	parallelize := func(fn func(start, end int)) {
		numWorkers := runtime.NumCPU()
		chunkSize := (numBlocks + numWorkers - 1) / numWorkers
		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			start := w * chunkSize
			end := start + chunkSize
			if end > numBlocks {
				end = numBlocks
			}
			if start >= numBlocks {
				break
			}
			wg.Add(1)
			go func(s, e int) {
				defer wg.Done()
				fn(s, e)
			}(start, end)
		}
		wg.Wait()
	}

	parallelize(func(start, end int) {
		for b := start; b < end; b++ {
			scale := t.scales[b]
			zero := t.zeros[b]
			blockStart := b * blockSize
			dataIdx := b * blockSize / 2
			for i := 0; i < blockSize; i += 2 {
				if dataIdx >= len(t.data) {
					break
				}
				qb := t.data[dataIdx]
				val0 := int8(qb&0x0F) - 8
				val1 := int8((qb>>4)&0x0F) - 8
				out[blockStart+i] = float32(val0)*scale + zero
				if blockStart+i+1 < len(out) {
					out[blockStart+i+1] = float32(val1)*scale + zero
				}
				dataIdx++
			}
		}
	})
}

func QuantizeFloat32(data []float32, rows, cols int, qtype QuantizationType, blockSize int) *QuantizedTensor {
	qt := &QuantizedTensor{
		rows:  rows,
		cols:  cols,
		qtype: qtype,
	}

	numBlocks := (rows*cols + blockSize - 1) / blockSize
	qt.scales = make([]float32, numBlocks)
	qt.zeros = make([]float32, numBlocks)

	if qtype == QuantInt8 {
		qt.data = make([]byte, rows*cols)
	} else {
		qt.data = make([]byte, (rows*cols+1)/2)
	}

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}

		var minVal, maxVal float32 = math.MaxFloat32, -math.MaxFloat32
		for i := start; i < end; i++ {
			if data[i] < minVal {
				minVal = data[i]
			}
			if data[i] > maxVal {
				maxVal = data[i]
			}
		}

		rangeVal := maxVal - minVal
		if rangeVal < 1e-6 {
			qt.scales[b] = 1.0
			qt.zeros[b] = 0.0
		} else {
			if qtype == QuantInt8 {
				qt.scales[b] = rangeVal / 255.0
				qt.zeros[b] = minVal
				for i := start; i < end; i++ {
					qt.data[i] = byte((data[i] - minVal) / qt.scales[b])
				}
			} else {
				qt.scales[b] = rangeVal / 15.0
				qt.zeros[b] = minVal
				for i := start; i < end; i += 2 {
					idx := i - start
					v0 := byte(((data[i] - minVal) / qt.scales[b]) + 0.5)
					var v1 byte
					if i+1 < end {
						v1 = byte(((data[i+1] - minVal) / qt.scales[b]) + 0.5)
					}
					qt.data[idx/2] = v0 | (v1 << 4)
				}
			}
		}
	}

	return qt
}
