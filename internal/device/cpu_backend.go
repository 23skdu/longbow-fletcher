package device

import (
	"log"
	"math"

	"sync"

	"github.com/23skdu/longbow-fletcher/internal/simd"
	"gonum.org/v1/gonum/mat"
)

// ensure interface compliance
var _ Backend = (*CPUBackend)(nil)
var _ Tensor = (*CPUTensor)(nil)

type CPUBackend struct{
	pool sync.Pool
}

func NewCPUBackend() *CPUBackend {
	return &CPUBackend{}
}

func (b *CPUBackend) Name() string {
	return "CPU"
}

func (b *CPUBackend) NewTensor(r, c int, data []float64) Tensor {
	return &CPUTensor{
		mat: mat.NewDense(r, c, data),
		backend: b,
	}
}


func (b *CPUBackend) GetTensor(r, c int) Tensor {
	// GetTensor is for getting a zero-initialized tensor from the pool.
	// It's essentially NewTensor(r, c, nil) but with pooling logic.
	return b.NewTensor(r, c, nil)
}

func (b *CPUBackend) PutTensor(t Tensor) {
	ct, ok := t.(*CPUTensor)
	if !ok {
		return // Don't pool foreign tensors
	}
	// Clear the tensor before putting it back to avoid holding references
	// and to ensure it's zero-initialized when retrieved by GetTensor.
	data := ct.mat.RawMatrix().Data
	for i := range data {
		data[i] = 0
	}
	b.pool.Put(ct)
}

func (b *CPUBackend) Synchronize() {
	// CPU is always synchronous
}

type CPUTensor struct {
	mat *mat.Dense // Use gonum/mat for operations
	backend *CPUBackend
}

func (t *CPUTensor) Dims() (int, int) {
	return t.mat.Dims()
}

func (t *CPUTensor) At(i, j int) float64 {
	return t.mat.At(i, j)
}

func (t *CPUTensor) Set(i, j int, v float64) {
	t.mat.Set(i, j, v)
}

func (t *CPUTensor) Copy(from Tensor) {
	ft, ok := from.(*CPUTensor)
	if !ok {
		// Fallback for cross-device copy (slow)
		// Or panic? For now, we only support same-device copy in basics
		log.Panic("Copying between different backends not yet supported directly")
	}
	t.mat.Copy(ft.mat)
}

func (t *CPUTensor) Slice(i, k, j, l int) Tensor {
	// mat.Slice returns Matrix interface, need to cast or access
	// But mat.Dense.Slice returns *mat.Dense (or compatible) usually? 
	// Actually Dense.Slice(i, k, j, l) returns Matrix, but we know implementation.
	// Wait, Dense.Slice creates a NEW view.
	view := t.mat.Slice(i, k, j, l).(*mat.Dense)
	return &CPUTensor{mat: view}
}

func (t *CPUTensor) T() Tensor {
	// mat.Dense T() returns Matrix.
	// NOTE: gonum T() is a view, but our Tensor interface implies it behaves like a Tensor.
	// Operations on T() might be limited in Gonum if it's just a transpose view?
	// Actually, for matmul it works. But for "in-place" edits it might not support Set?
	// Gonum's Transpose type allows At, but Set is not guaranteed.
	// For simplicty, let's create a *copy* if we need a mutable tensor, or wrap a Transpose view?
	// But our interface demands Mutable Tensor capabilities (Set, Scale).
	// A Transpose view in Gonum is read-only for Set usually?
	// Check gonum docs: T() returns Matrix. 
	// If we need a mutable transpose, we might need a shadow Copy or deal with it.
	// However, for matmul A * B.T(), we usually just want the view.
	// Let's implement T as a Copy for safely mutable, OR support read-only views.
	// Given performance is key, copying is bad.
	// But `device.Tensor` assumes mutable.
	// Let's Panic on T() until we verify usage. 
	// Actually, in Bert, we utilize T() for MatMul input: `keyLayer.T()`.
	// We only use it for reading in MatMul. 
	// So we can support a "ViewTensor" that panics on writes?
	// Simplest for CPU backend now: return a new CPUTensor that IS the transpose (copy).
	// Wait, making a copy for every attention head is SLOW.
	// We should support "view" semantics or update interface to allow R/O Tensors.
	
	// Better approach for CPU backend: 
	// Allow `Mul` to take `Tensor` and do type assertion to `mat.Matrix`.
	// This lets us pass `t.mat.T()` directly to `Mul`.
	// But `T()` must return `Tensor`.
	
	// Hack: We return a *CPUTensor but the internal mat is a Transpose? 
	// mat.Dense doesn't wrap Transpose.
	// Let's rely on type assertion in Mul.
	// But T() return type is Tensor. 
	// Let's make T() create a Copy for now (correctness > perf for initial refactor),
	// OR we introduce `TransposeView` type.
	
	// REVISION: The usage in Bert is `attentionScores.Mul(queryLayer, keyLayerT)`.
	// Since `Mul` is on the receiver, `keyLayerT` is an argument.
	// If we return a wrapper around `mat.Transpose`, `Mul` (gonum) accepts `mat.Matrix`.
	// So we need a `CPUViewTensor` that wraps `mat.Matrix` (read-only?).
	
	r, c := t.mat.Dims()
	out := mat.NewDense(c, r, nil)
	out.Copy(t.mat.T())
	return &CPUTensor{mat: out}
}

func (t *CPUTensor) Mul(a, b Tensor) {
	// a and b must be CPUTensors
	at, ok1 := a.(*CPUTensor)
	bt, ok2 := b.(*CPUTensor)
	
	if !ok1 || !ok2 {
		log.Panic("Mixed backend Mul not supported")
	}
	
	t.mat.Mul(at.mat, bt.mat)
}

func (t *CPUTensor) Add(other Tensor) {
	ot, ok := other.(*CPUTensor)
	if !ok {
		log.Panic("Mixed backend Add not supported")
	}
	// Use our SIMD optimized AddInPlace
	simd.VecAdd(t.mat.RawMatrix().Data, ot.mat.RawMatrix().Data)
}

func (t *CPUTensor) AddScalar(val float64) {
	data := t.mat.RawMatrix().Data
	// No SIMD for scalar add yet, just loop
	for i := range data {
		data[i] += val
	}
}

func (t *CPUTensor) AddBias(bias Tensor) {
	r, c := t.Dims()
	br, bc := bias.Dims()
	
	// Bias should be a vector of length c.
	// Either 1xc or cx1.
	if r == 1 && c == bc {
		// Single row matching
	} else if bc != c {
		panic("AddBias dimension mismatch")
	}
	
	// Get raw slice from bias tensor (assumed CPU)
	// We could use bias.Data() if available, or ToHost() copy.
	// Since we are in CPU backend, assume bias is CPUTensor or compatible.
	var biasData []float64
	if ct, ok := bias.(*CPUTensor); ok {
		biasData = ct.mat.RawMatrix().Data
	} else {
		// Fallback for cross-device (shouldn't happen in pure CPU run)
		biasData = bias.ToHost()
	}
	
	if len(biasData) != c {
		// Check if it's 1xM or Mx1
		if br * bc != c {
			panic("AddBias: bias length mismatch")
		}
	}

	data := t.mat.RawMatrix().Data
	// Add bias to each row
	for i := 0; i < r; i++ {
		row := data[i*c : (i+1)*c]
		simd.VecAdd(row, biasData)
	}
}

func (t *CPUTensor) Scale(val float64) {
	// Use scaler scaling? Or simple loop?
	// mat.Scale does this.
	t.mat.Scale(val, t.mat)
}

func (t *CPUTensor) Gather(indices []int) Tensor {
	r, c := t.Dims()
	
	outData := make([]float64, len(indices)*c)
	tData := t.mat.RawMatrix().Data
	
	// parallel := len(indices) * c > 10000 
	
	// If parallel, use goroutines? Simple loop for now.
	// Actually for large embeddings, parallel copy is good.
	// But let's stick to simple copy.
	
	for i, idx := range indices {
		if idx < 0 || idx >= r {
			panic("Gather index out of bounds")
		}
		// Copy row idx to row i
		copy(outData[i*c : (i+1)*c], tData[idx*c : (idx+1)*c])
	}
	
	// Create new tensor
	return t.backend.NewTensor(len(indices), c, outData)
}

func (t *CPUTensor) Softmax() {
	r, _ := t.mat.Dims()
	for i := 0; i < r; i++ {
		row := t.mat.RawRowView(i)
		simd.SoftmaxFast(row)
	}
}

func (t *CPUTensor) Gelu() {
	simd.GeluFast(t.mat.RawMatrix().Data)
}

func (t *CPUTensor) Tanh() {
	data := t.mat.RawMatrix().Data
	for i, v := range data {
		data[i] = simd.TanhFast(v)
	}
}

func (t *CPUTensor) LayerNorm(gamma, beta Tensor, eps float64) {
	gt, ok1 := gamma.(*CPUTensor)
	bt, ok2 := beta.(*CPUTensor)
	if !ok1 || !ok2 {
		log.Panic("Mixed backend LayerNorm not supported")
	}
	
	// Access raw data
	data := t.mat.RawMatrix().Data
	gammaData := gt.mat.RawMatrix().Data
	betaData := bt.mat.RawMatrix().Data
	
	r, c := t.mat.Dims()
	
	// Assuming gamma/beta are size c
	if len(gammaData) < c || len(betaData) < c {
		log.Panic("LayerNorm params dim mismatch")
	}
	
	for i := 0; i < r; i++ {
		rowStart := i * c
		row := data[rowStart : rowStart+c]
		
		// 1. Calculate Mean
		var sum float64
		for _, v := range row {
			sum += v
		}
		mean := sum / float64(c)
		
		// 2. Calculate Variance
		var varSum float64
		for _, v := range row {
			diff := v - mean
			varSum += diff * diff
		}
		variance := varSum / float64(c)
		invStd := 1.0 / math.Sqrt(variance + eps)
		
		// 3. Normalize and Scale/Shift
		for j := 0; j < c; j++ {
			row[j] = (row[j] - mean) * invStd * gammaData[j] + betaData[j]
		}
	}
}

func (t *CPUTensor) Linear(input, weight, bias Tensor) Tensor {
	// 1. MatMul: result = input * weight
	// Output dims: input.rows x weight.cols
	r, _ := input.Dims()
	_, wc := weight.Dims()
	
	result := t.backend.NewTensor(r, wc, nil)
	result.Mul(input, weight)
	
	// 2. Add Bias
	if bias != nil {
		result.AddBias(bias)
	}
	
	return result
}

func (t *CPUTensor) LinearActivation(input, weight, bias Tensor, activation ActivationType) Tensor {
	// 1. Linear
	result := t.Linear(input, weight, bias)
	
	// 2. Activation
	switch activation {
	case ActivationGELU:
		result.Gelu()
	case ActivationTanh:
		result.Tanh()
	case ActivationSoftmax:
		result.Softmax()
	case ActivationIdentity:
		// No-op
	}
	
	return result
}



func (t *CPUTensor) Attention(q, k, v Tensor, batchSize, seqLen int, scale float64) Tensor {
	qt := q.(*CPUTensor)
	kt := k.(*CPUTensor)
	vt := v.(*CPUTensor)
	
	r, c := qt.Dims()
	if r != batchSize*seqLen {
		panic("Attention: dims mismatch")
	}
	
	result := t.backend.NewTensor(r, c, nil)
	rst := result.(*CPUTensor)
	
	// Buffers for intermediate per-sequence calculations
	// To avoid allocs in loop, we could reuse, but simple loop is fine for now.
	// We need a temporary score matrix (seqLen x seqLen)
	
	for i := 0; i < batchSize; i++ {
		start := i * seqLen
		// end unused
		
		// Slices (Views)
		// Use Slice method to handle indexing logic safely
		qSlice := qt.Slice(start, start+seqLen, 0, c).(*CPUTensor).mat
		kSlice := kt.Slice(start, start+seqLen, 0, c).(*CPUTensor).mat
		// vSlice used in Context calculation, also needs to be *mat.Dense
		vSlice := vt.Slice(start, start+seqLen, 0, c).(*CPUTensor).mat
		outSlice := rst.Slice(start, start+seqLen, 0, c).(*CPUTensor).mat
		
		// qSlice * kSlice.T
		var scores mat.Dense
		scores.Mul(qSlice, kSlice.T())
		
		// Scale
		scores.Scale(scale, &scores)
		
		// Softmax row-wise
		// Access raw data of scores
		rows, _ := scores.Dims()
		for r := 0; r < rows; r++ {
			row := scores.RawRowView(r)
			simd.SoftmaxFast(row)
		}
		
		// Context = Scores * V
		outSlice.Mul(&scores, vSlice)
	}
	
	return result
}

func (t *CPUTensor) ToHost() []float64 {
	// For CPU, we just return a copy of the data? 
	// Or raw data? "ToHost" implies transfer. 
	// To be safe and immutable-ish, copy.
	// But for performance, maybe raw? 
	// Let's return copy to be safe.
	// Actually, `mat.RawMatrix().Data` is the slice.
	// Given this is an interface for mostly GPU sync, returning the slice is fine.
	// But GPU would return a copy.
	// Let's return a copy to unify behavior.
	data := t.mat.RawMatrix().Data
	dst := make([]float64, len(data))
	copy(dst, data)
	return dst
}

func (t *CPUTensor) Data() []float64 {
    return t.mat.RawMatrix().Data
}

func (t *CPUTensor) CopyFromFloat64(data []float64) {
	dst := t.mat.RawMatrix().Data
	if len(data) != len(dst) {
		panic("CopyFromFloat64: size mismatch")
	}
	copy(dst, data)
}
