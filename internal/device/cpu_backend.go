package device

import (
	"log"
	"math"
	"sync"
	"runtime"

	"github.com/23skdu/longbow-fletcher/internal/simd"
)

// ensure interface compliance
var _ Backend = (*CPUBackend)(nil)
var _ Tensor = (*CPUTensor)(nil)

// numWorkers defines the default parallelism for CPU operations
var numWorkers = runtime.NumCPU()

type CPUBackend struct{
	pool sync.Pool
}

func NewCPUBackend() *CPUBackend {
	return &CPUBackend{
		pool: sync.Pool{
			New: func() interface{} {
				// Initialize a new CPUTensor
				return &CPUTensor{}
			},
		},
	}
}

func (b *CPUBackend) Name() string {
	return "CPU"
}

func (b *CPUBackend) NewTensor(r, c int, data []float32) Tensor {
	size := r * c
	t := &CPUTensor{
		backend: b,
		rows:    r,
		cols:    c,
	}
	
	if data == nil {
		t.data = make([]float32, size)
	} else {
		if len(data) != size {
			panic("NewTensor: provided data length does not match dimensions")
		}
		t.data = make([]float32, size)
		copy(t.data, data)
	}
	
	return t
}

func (b *CPUBackend) GetTensor(r, c int) Tensor {
	// Try to get from pool
	v := b.pool.Get()
	ct, ok := v.(*CPUTensor)
	if !ok || ct == nil {
		ct = &CPUTensor{}
	}

	// Initialize/reset the tensor
	ct.backend = b
	ct.rows = r
	ct.cols = c
	ct.trans = false
	size := r * c
	if cap(ct.data) < size {
		ct.data = make([]float32, size)
	} else {
		ct.data = ct.data[:size] // Reslice to correct size
		// Zero-initialize
		for i := range ct.data {
			ct.data[i] = 0.0
		}
	}
	return ct
}

func (b *CPUBackend) PutTensor(t Tensor) {
	ct, ok := t.(*CPUTensor)
	if !ok {
		return // Don't pool foreign tensors
	}
	
	ct.rows = 0
	ct.cols = 0
	ct.trans = false
	// Data is zeroed when retrieved by GetTensor
	b.pool.Put(ct)
}

func (b *CPUBackend) Synchronize() {
	// CPU is always synchronous
}

type CPUTensor struct {
	backend *CPUBackend
	data    []float32
	rows    int
	cols    int
	trans   bool // Transposed view flag
}

func (t *CPUTensor) Dims() (int, int) {
	if t.trans {
		return t.cols, t.rows
	}
	return t.rows, t.cols
}

func (t *CPUTensor) At(i, j int) float32 {
	if t.trans {
		// Logical (i, j) -> Physical (j, i)
		return t.data[j*t.cols+i]
	}
	return t.data[i*t.cols+j]
}

func (t *CPUTensor) Set(i, j int, v float32) {
	if t.trans {
		t.data[j*t.cols+i] = v
	} else {
		t.data[i*t.cols+j] = v
	}
}

func (t *CPUTensor) Data() []float32 {
	// If transposed, data is not contiguous in logical order
	if t.trans {
		return nil
	}
	return t.data
}

func (t *CPUTensor) ToHost() []float32 {
	if t.trans {
		// Need to physical copy to respect transpose
		rows, cols := t.Dims()
		out := make([]float32, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[i*cols+j] = t.At(i, j)
			}
		}
		return out
	}
	
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return out
}

func (t *CPUTensor) CopyFromFloat32(data []float32) {
	if len(data) != len(t.data) {
		panic("Size mismatch")
	}
	copy(t.data, data)
}

func (t *CPUTensor) Copy(from Tensor) {
	ft, ok := from.(*CPUTensor)
	if !ok {
		log.Panic("Copying between different backends not yet supported directly")
	}
	
	tr, tc := t.Dims()
	fr, fc := ft.Dims()

	if tr != fr || tc != fc {
		log.Panicf("Copy: dimension mismatch. Target: %dx%d, Source: %dx%d", tr, tc, fr, fc)
	}

	if !t.trans && !ft.trans {
		// Both are non-transposed, direct copy of underlying data
		copy(t.data, ft.data)
	} else {
		// One or both are transposed, element by element copy
		for i := 0; i < tr; i++ {
			for j := 0; j < tc; j++ {
				t.Set(i, j, ft.At(i, j))
			}
		}
	}
}

func (t *CPUTensor) Slice(i, k, j, l int) Tensor {
	sliceRows := k - i
	sliceCols := l - j

	if sliceRows <= 0 || sliceCols <= 0 {
		panic("Slice: invalid dimensions")
	}

	// This is a copy, not a view.
	out := t.backend.NewTensor(sliceRows, sliceCols, nil)
	for rowIdx := 0; rowIdx < sliceRows; rowIdx++ {
		for colIdx := 0; colIdx < sliceCols; colIdx++ {
			out.Set(rowIdx, colIdx, t.At(i+rowIdx, j+colIdx))
		}
	}
	return out
}

func (t *CPUTensor) T() Tensor {
	return &CPUTensor{
		backend: t.backend,
		data:    t.data, // Share data
		rows:    t.rows,
		cols:    t.cols,
		trans:   !t.trans, // Toggle transpose state
	}
}

func (t *CPUTensor) Mul(a, b Tensor) {
	// a and b must be CPUTensors
	ma, ok1 := a.(*CPUTensor)
	mb, ok2 := b.(*CPUTensor)
	
	if !ok1 || !ok2 {
		log.Panic("Mixed backend Mul not supported")
	}

	ar, ac := ma.Dims()
	br, bc := mb.Dims()

	if ac != br {
		log.Panicf("Mul: dimension mismatch. A cols (%d) != B rows (%d)", ac, br)
	}

	tr, tc := t.Dims()
	if tr != ar || tc != bc {
		log.Panicf("Mul: result tensor dimension mismatch. Expected %dx%d, got %dx%d", ar, bc, tr, tc)
	}
	
	common := ac // or br

	// Parallel MatMul
	var wg sync.WaitGroup
	rowsPerWorker := (ar + numWorkers - 1) / numWorkers
	
	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if startRow >= ar {
			break
		}
		if endRow > ar {
			endRow = ar
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			
			// For each row in result
			for i := start; i < end; i++ {
				// C[i, j] = A[i, :] * B[:, j]
				
				// Get row A[i]
				var rowA []float32
				if ma.trans {
					rowA = make([]float32, common)
					for k := 0; k < common; k++ {
						rowA[k] = ma.At(i, k)
					}
				} else {
					startA := i * ma.cols
					rowA = ma.data[startA : startA+ma.cols]
				}
				
				for j := 0; j < bc; j++ {
					// Get col j of B
					var colB []float32
					if mb.trans {
						startB := j * mb.cols
						colB = mb.data[startB : startB+mb.cols]
					} else {
						colB = make([]float32, common)
						for k := 0; k < common; k++ {
							colB[k] = mb.At(k, j)
						}
					}
					
					val := simd.DotProduct(rowA, colB)
					t.Set(i, j, val)
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}

func (t *CPUTensor) Add(other Tensor) {
	ot, ok := other.(*CPUTensor)
	if !ok {
		log.Panic("Mixed backend Add not supported")
	}

	tr, tc := t.Dims()
	or, oc := ot.Dims()

	if tr != or || tc != oc {
		log.Panicf("Add: dimension mismatch. Target: %dx%d, Other: %dx%d", tr, tc, or, oc)
	}

	if !t.trans && !ot.trans {
		simd.VecAdd(t.data, ot.data)
	} else {
		for i := 0; i < tr; i++ {
			for j := 0; j < tc; j++ {
				t.Set(i, j, t.At(i, j) + ot.At(i, j))
			}
		}
	}
}

func (t *CPUTensor) AddScalar(val float32) {
	for i := range t.data {
		t.data[i] += val
	}
}

func (t *CPUTensor) AddBias(bias Tensor) {
	bt, ok := bias.(*CPUTensor)
	if !ok { panic("Mixed backend AddBias") }
	
	r, c := t.Dims()
	br, bc := bias.Dims()
	
	if br != 1 && bc != 1 {
		panic("AddBias: bias must be a vector (1xN or Nx1)")
	}
	
	var biasData []float32
	if bt.trans {
		biasData = make([]float32, c)
		if br == 1 { // bias is 1xc
			for i := 0; i < c; i++ {
				biasData[i] = bt.At(0, i)
			}
		} else { // bias is cx1
			for i := 0; i < c; i++ {
				biasData[i] = bt.At(i, 0)
			}
		}
	} else {
		biasData = bt.data
	}
	
	if len(biasData) != c {
		panic("AddBias: bias length mismatch with tensor columns")
	}

	if t.trans {
		log.Panic("AddBias not supported on transposed tensor views directly")
	}

	data := t.data
	for i := 0; i < r; i++ {
		row := data[i*c : (i+1)*c]
		simd.VecAdd(row, biasData)
	}
}

func (t *CPUTensor) Scale(val float32) {
	for i := range t.data {
		t.data[i] *= val
	}
}

func (t *CPUTensor) Gather(indices []int) Tensor {
	r, c := t.Dims()
	outData := make([]float32, len(indices)*c)
	
	for i, idx := range indices {
		if idx < 0 || idx >= r {
			panic("Gather index out of bounds")
		}
		for j := 0; j < c; j++ {
			outData[i*c+j] = t.At(idx, j)
		}
	}
	
	return t.backend.NewTensor(len(indices), c, outData)
}

func (t *CPUTensor) Softmax() {
	if t.trans {
		panic("Softmax on transposed")
	}
	r, c := t.Dims()
	for i := 0; i < r; i++ {
		rowStart := i * c
		row := t.data[rowStart : rowStart+c]
		simd.SoftmaxFast(row)
	}
}

func (t *CPUTensor) Gelu() {
	if t.trans {
		log.Panic("Gelu not supported on transposed tensor views directly")
	}
	simd.GeluFast(t.data)
}

func (t *CPUTensor) Tanh() {
	if t.trans {
		log.Panic("Tanh not supported on transposed tensor views directly")
	}
	data := t.data
	for i, v := range data {
		data[i] = simd.TanhFast(v)
	}
}

func (t *CPUTensor) LayerNorm(gamma, beta Tensor, eps float32) {
	gt, ok1 := gamma.(*CPUTensor)
	bt, ok2 := beta.(*CPUTensor)
	if !ok1 || !ok2 { panic("Mixed backend LN") }
	
	if t.trans {
		log.Panic("LayerNorm not supported on transposed tensor views directly")
	}
	data := t.data

	var gammaData, betaData []float32
	_, gc := gt.Dims()
	_, bc := bt.Dims()

	if gt.trans {
		gammaData = make([]float32, gc)
		for i := 0; i < gc; i++ {
			gammaData[i] = gt.At(0, i)
		}
	} else {
		gammaData = gt.data
	}

	if bt.trans {
		betaData = make([]float32, bc)
		for i := 0; i < bc; i++ {
			betaData[i] = bt.At(0, i)
		}
	} else {
		betaData = bt.data
	}
	
	r, c := t.Dims()
	
	if len(gammaData) < c || len(betaData) < c {
		log.Panic("LayerNorm params dim mismatch")
	}
	
	for i := 0; i < r; i++ {
		rowStart := i * c
		row := data[rowStart : rowStart+c]
		
		var sum float32
		for _, v := range row {
			sum += v
		}
		mean := sum / float32(c)
		
		var varSum float32
		for _, v := range row {
			diff := v - mean
			varSum += diff * diff
		}
		variance := varSum / float32(c)
		invStd := 1.0 / float32(math.Sqrt(float64(variance + eps)))
		
		for j := 0; j < c; j++ {
			row[j] = (row[j] - mean) * invStd * gammaData[j] + betaData[j]
		}
	}
}

func (t *CPUTensor) Linear(input, weight, bias Tensor) Tensor {
	r, _ := input.Dims()
	_, wc := weight.Dims()
	
	result := t.backend.GetTensor(r, wc)
	result.Mul(input, weight)
	
	if bias != nil {
		result.AddBias(bias)
	}
	
	return result
}

func (t *CPUTensor) LinearActivation(input, weight, bias Tensor, activation ActivationType) Tensor {
	result := t.Linear(input, weight, bias)
	
	switch activation {
	case ActivationGELU:
		result.Gelu()
	case ActivationTanh:
		result.Tanh()
	case ActivationSoftmax:
		result.Softmax()
	case ActivationSwiGLU:
		// Result of Linear is (N, 2 * interSize)
		r, c := result.Dims()
		interSize := c / 2
		output := t.backend.GetTensor(r, interSize)
		ot := output.(*CPUTensor)
		rt := result.(*CPUTensor)
		
		for n := 0; n < r; n++ {
			for i := 0; i < interSize; i++ {
				x := rt.data[n*c + i]
				y := rt.data[n*c + i + interSize]
				// Swish(x) = x * sigmoid(x)
				swishX := x / (1.0 + float32(math.Exp(float64(-x))))
				ot.data[n*interSize + i] = swishX * y
			}
		}
		t.backend.PutTensor(result)
		return output
	case ActivationIdentity:
		// No-op
	}
	
	return result
}

func (t *CPUTensor) Attention(q, k, v Tensor, batchSize, seqLen int, scale float32) Tensor {
	qt := q.(*CPUTensor)
	kt := k.(*CPUTensor)
	vt := v.(*CPUTensor)
	
	r, c := qt.Dims()
	if r != batchSize*seqLen {
		panic("Attention: dims mismatch")
	}
	
	result := t.backend.NewTensor(r, c, nil)
	rst := result.(*CPUTensor)
	
	var wg sync.WaitGroup
	workers := numWorkers
	if batchSize < workers {
		workers = batchSize
	}
	
	itemsPerWorker := (batchSize + workers - 1) / workers
	
	for w := 0; w < workers; w++ {
		startBatch := w * itemsPerWorker
		endBatch := startBatch + itemsPerWorker
		if endBatch > batchSize {
			endBatch = batchSize
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			
			scoresBuf := make([]float32, seqLen*seqLen)
			
			for i := start; i < end; i++ {
				offset := i * seqLen
				
				for rQ := 0; rQ < seqLen; rQ++ {
					qIdx := (offset + rQ) * c
					qRow := qt.data[qIdx : qIdx+c]
					
					for rK := 0; rK < seqLen; rK++ {
						kIdx := (offset + rK) * c
						kRow := kt.data[kIdx : kIdx+c]
						
						dot := simd.DotProduct(qRow, kRow)
						scoresBuf[rQ*seqLen + rK] = dot * scale
					}
				}
				
				for rS := 0; rS < seqLen; rS++ {
					rowIdx := rS * seqLen
					simd.SoftmaxFast(scoresBuf[rowIdx : rowIdx+seqLen])
				}
				
				for rS := 0; rS < seqLen; rS++ {
					outIdx := (offset + rS) * c
					outRow := rst.data[outIdx : outIdx+c]
					
					for z := 0; z < c; z++ {
						outRow[z] = 0
					}
					
					scoresRowIdx := rS * seqLen
					scoresRow := scoresBuf[scoresRowIdx : scoresRowIdx+seqLen]
					
					for k := 0; k < seqLen; k++ {
						score := scoresRow[k]
						vIdx := (offset + k) * c
						vRow := vt.data[vIdx : vIdx+c]
						
						simd.VecAddScaled(outRow, vRow, score)
					}
				}
			}
		}(startBatch, endBatch)
	}
	wg.Wait()
	
	return result
}

func (t *CPUTensor) ApplyRoPE(batchSize, seqLen, numHeads, headDim int) {
	if t.trans {
		panic("ApplyRoPE on transposed")
	}
	
	totalRows := batchSize * seqLen
	var wg sync.WaitGroup
	rowsPerWorker := (totalRows + numWorkers - 1) / numWorkers
	
	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if startRow >= totalRows {
			break
		}
		if endRow > totalRows {
			endRow = totalRows
		}
		
		wg.Add(1)
		go func(sRow, eRow int) {
			defer wg.Done()
			for r := sRow; r < eRow; r++ {
				seqIdx := r % seqLen
				rowOffset := r * (numHeads * headDim)
				
				for h := 0; h < numHeads; h++ {
					headOffset := rowOffset + h * headDim
					
					for i := 0; i < headDim/2; i++ {
						theta := float64(seqIdx) * math.Pow(10000.0, -2.0*float64(i)/float64(headDim))
						cosTheta := float32(math.Cos(theta))
						sinTheta := float32(math.Sin(theta))
						
						x1 := t.data[headOffset + i]
						x2 := t.data[headOffset + headDim/2 + i]
						
						t.data[headOffset + i] = x1*cosTheta - x2*sinTheta
						t.data[headOffset + headDim/2 + i] = x1*sinTheta + x2*cosTheta
					}
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()
}
