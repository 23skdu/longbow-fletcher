//go:build linux && cuda

package device

/*
#cgo LDFLAGS: -L. -lcuda_fletcher -lcudart -lcublas
#include "cuda_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Check interface compliance
var _ Backend = (*CudaBackend)(nil)
var _ Tensor = (*CudaTensor)(nil)

type CudaBackend struct {
	ctx     C.CudaContextRef
	useFP16 bool
}

func NewCudaBackend() *CudaBackend {
	ctx := C.Cuda_Init()
	if ctx == nil {
		panic("Failed to initialize CUDA backend")
	}
	return &CudaBackend{ctx: ctx, useFP16: false}
}

func NewCudaBackendFP16() *CudaBackend {
	ctx := C.Cuda_Init()
	if ctx == nil {
		panic("Failed to initialize CUDA backend")
	}
	return &CudaBackend{ctx: ctx, useFP16: true}
}

func (b *CudaBackend) Name() string {
	if b.useFP16 {
		return "CUDA-MatX-FP16"
	}
	return "CUDA-MatX"
}

func (b *CudaBackend) NewTensor(r, c int, data []float32) Tensor {
	t := b.GetTensor(r, c)
	if data != nil {
		t.CopyFromFloat32(data)
	}
	return t
}

func (b *CudaBackend) GetTensor(r, c int) Tensor {
	size := r * c
	var sizeBytes int
	if b.useFP16 {
		sizeBytes = size * 2
	} else {
		sizeBytes = size * 4
	}

	buf := C.Cuda_Alloc(b.ctx, C.int(sizeBytes))
	if buf == nil {
		panic("Failed to allocate CUDA memory")
	}

	t := &CudaTensor{
		backend:   b,
		buf:       buf,
		rows:      r,
		cols:      c,
		sizeBytes: sizeBytes,
	}

	runtime.SetFinalizer(t, func(t *CudaTensor) {
		C.Cuda_FreeBuffer(t.backend.ctx, t.buf)
	})

	return t
}

func (b *CudaBackend) PutTensor(t Tensor) {
	// For now, simplicity: let GC handle it or add a pool later
}

func (b *CudaBackend) Synchronize() {
	C.Cuda_Synchronize(b.ctx)
}

func (b *CudaBackend) DeviceCount() int {
	return int(C.Cuda_GetDeviceCount())
}

func (b *CudaBackend) SetDevice(index int) {
	C.Cuda_SetDevice(b.ctx, C.int(index))
}

func (b *CudaBackend) GetVRAMUsage() (int64, int64) {
	// TODO: Implement cudaMemGetInfo
	return 0, 0
}

type CudaTensor struct {
	backend   *CudaBackend
	buf       C.CudaBufferRef
	rows      int
	cols      int
	sizeBytes int
}

func (t *CudaTensor) Dims() (int, int) {
	return t.rows, t.cols
}

func (t *CudaTensor) At(i, j int) float32 {
	// Slow path for debugging
	var val float32
	C.Cuda_CopyToHost(t.buf, C.int((i*t.cols+j)*4), unsafe.Pointer(&val), 4)
	return val
}

func (t *CudaTensor) Set(i, j int, v float32) {
	if t.backend.useFP16 {
		f16 := Float32ToFloat16(v)
		C.Cuda_CopyToDevice(t.buf, C.int((i*t.cols+j)*2), unsafe.Pointer(&f16), 2)
	} else {
		C.Cuda_CopyToDevice(t.buf, C.int((i*t.cols+j)*4), unsafe.Pointer(&v), 4)
	}
}

func (t *CudaTensor) Data() []float32 {
	return nil // Resident on GPU
}

func (t *CudaTensor) ToHost() []float32 {
	size := t.rows * t.cols
	
	if t.backend.useFP16 {
		raw16 := make([]uint16, size)
		C.Cuda_CopyToHost(t.buf, 0, unsafe.Pointer(&raw16[0]), C.int(t.sizeBytes))
		
		data := make([]float32, size)
		for i, h := range raw16 {
			data[i] = Float16ToFloat32(h)
		}
		return data
	}
	
	data := make([]float32, size)
	C.Cuda_CopyToHost(t.buf, 0, unsafe.Pointer(&data[0]), C.int(t.sizeBytes))
	return data
}

func (t *CudaTensor) CopyFromFloat32(data []float32) {
	if t.backend.useFP16 {
		size := len(data)
		f16 := make([]uint16, size)
		for i, v := range data {
			f16[i] = Float32ToFloat16(v)
		}
		C.Cuda_CopyToDevice(t.buf, 0, unsafe.Pointer(&f16[0]), C.int(t.sizeBytes))
	} else {
		C.Cuda_CopyToDevice(t.buf, 0, unsafe.Pointer(&data[0]), C.int(t.sizeBytes))
	}
}

func (t *CudaTensor) Copy(from Tensor) {
	ft := from.(*CudaTensor)
	C.Cuda_CopyToDevice(t.buf, 0, ft.buf, C.int(t.sizeBytes))
}

func (t *CudaTensor) Slice(i, k, j, l int) Tensor {
	panic("Slice not implemented for CUDA")
}

func (t *CudaTensor) T() Tensor {
	panic("Transpose view not implemented for CUDA (handled in MatMul)")
}

func (t *CudaTensor) Mul(a, b Tensor) {
	t.Linear(a, b, nil)
}

func (t *CudaTensor) Add(other Tensor) {
	panic("Add not implemented")
}

func (t *CudaTensor) AddScalar(val float32) {
	panic("AddScalar not implemented")
}

func (t *CudaTensor) Scale(val float32) {
	panic("Scale not implemented")
}

func (t *CudaTensor) AddBias(bias Tensor) {
	panic("AddBias not implemented separately (fused in Linear)")
}

func (t *CudaTensor) Softmax() {
	C.Cuda_Softmax(t.backend.ctx, t.buf, t.buf, C.int(t.rows), C.int(t.cols))
}

func (t *CudaTensor) Gelu() {
	t.LinearActivation(t, nil, nil, ActivationGELU) // Self-GELU
}

func (t *CudaTensor) Tanh() {
	panic("Tanh not implemented")
}

func (t *CudaTensor) LayerNorm(gamma, beta Tensor, eps float32) {
	gg := gamma.(*CudaTensor)
	bb := beta.(*CudaTensor)
	C.Cuda_LayerNorm(t.backend.ctx, t.buf, gg.buf, bb.buf, t.buf, C.int(t.rows), C.int(t.cols), C.float(eps))
}

func (t *CudaTensor) Gather(indices []int) Tensor {
	idxRows := len(indices)
	out := t.backend.GetTensor(idxRows, t.cols)
	
	// Copy indices to device (temp allocation)
	idxBuf := C.Cuda_Alloc(t.backend.ctx, C.int(idxRows*4))
	defer C.Cuda_FreeBuffer(t.backend.ctx, idxBuf)
	
	C.Cuda_CopyToDevice(idxBuf, 0, unsafe.Pointer(&indices[0]), C.int(idxRows*4))
	
	C.Cuda_Gather(t.backend.ctx, t.buf, idxBuf, out.(*CudaTensor).buf, C.int(idxRows), C.int(t.cols))
	return out
}

func (t *CudaTensor) Linear(input, weight, bias Tensor) Tensor {
	return t.LinearActivation(input, weight, bias, ActivationIdentity)
}

func (t *CudaTensor) LinearActivation(input, weight, bias Tensor, activation ActivationType) Tensor {
	in := input.(*CudaTensor)
	var w, b C.CudaBufferRef
	var wtCols int
	
	if weight != nil {
		w = weight.(*CudaTensor).buf
		_, wtCols = weight.Dims()
	} else {
		// If weight is nil, it's an element-wise activation on input
		wtCols = in.cols
	}
	
	if bias != nil {
		b = bias.(*CudaTensor).buf
	}

	C.Cuda_Linear_Fused(t.backend.ctx, in.buf, C.int(in.rows), C.int(in.cols), 
	                   w, C.int(wtCols), b, t.buf, C.int(activation))
	return t
}

func (t *CudaTensor) Attention(q, k, v Tensor, batchSize, seqLen int, scale float32) Tensor {
	qt := q.(*CudaTensor)
	kt := k.(*CudaTensor)
	vt := v.(*CudaTensor)
	// Hidden size is cols of q
	_, hiddenSize := qt.Dims()
	C.Cuda_Attention_Fused(
		t.backend.ctx,
		qt.buf,
		kt.buf,
		vt.buf,
		t.buf,
		C.int(batchSize),
		C.int(seqLen),
		C.int(hiddenSize),
		C.float(scale),
	)
	return t
}

func (t *CudaTensor) ApplyRoPE(batchSize, seqLen, numHeads, headDim int) {
	C.Cuda_ApplyRoPE(t.backend.ctx, t.buf, C.int(batchSize), C.int(seqLen), C.int(numHeads), C.int(headDim))
}

func (t *CudaTensor) ExtractTo(destination [][]float32, startRow int) {
	// For CUDA, we'll use a simple host copy for now. 
	// Future optimization: pinned memory and async streaming.
	data := t.ToHost()
	r, c := t.rows, t.cols
	for i := 0; i < r; i++ {
		row := make([]float32, c)
		copy(row, data[i*c:(i+1)*c])
		destination[startRow+i] = row
	}
}

func (t *CudaTensor) ExtractToFlat(dest []float32, start int) {
	data := t.ToHost()
	copy(dest[start:], data)
}
