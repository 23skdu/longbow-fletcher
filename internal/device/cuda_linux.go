//go:build linux && cuda

package device

/*
#cgo LDFLAGS: -L. -lcuda_fletcher -lcudart -lcublas
#include "cuda_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
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

func (b *CudaBackend) NewTensorWithType(r, c int, dtype DataType, data []float32) Tensor {
	// Handle FP16 vs FP32 based on dtype, not backend's default
	origFP16 := b.useFP16
	if dtype == Float16 {
		b.useFP16 = true
	} else {
		b.useFP16 = false
	}
	t := b.GetTensor(r, c)
	b.useFP16 = origFP16 // restore

	if data != nil {
		t.CopyFromFloat32(data)
	}
	return t
}

func (b *CudaBackend) GetTensor(r, c int) Tensor {
	size := r * c
	var sizeBytes int
	var dtype DataType
	if b.useFP16 {
		sizeBytes = size * 2
		dtype = Float16
	} else {
		sizeBytes = size * 4
		dtype = Float32
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
		dtype:     dtype,
	}

	runtime.SetFinalizer(t, func(t *CudaTensor) {
		C.Cuda_FreeBuffer(t.backend.ctx, t.buf)
	})

	return t
}

func (b *CudaBackend) GetTensorOfType(r, c int, dtype DataType) Tensor {
	size := r * c
	elemSize := dtypeSize(dtype)
	sizeBytes := size * elemSize

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
		dtype:     dtype,
	}

	runtime.SetFinalizer(t, func(t *CudaTensor) {
		C.Cuda_FreeBuffer(t.backend.ctx, t.buf)
	})

	return t
}

func dtypeSize(dtype DataType) int {
	switch dtype {
	case Float16:
		return 2
	case Float32, Int32, Uint32:
		return 4
	case Float64, Int64, Uint64:
		return 8
	case Int8, Uint8:
		return 1
	default:
		return 4
	}
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
	var free, total int64
	C.Cuda_GetMemoryInfo(b.ctx, (*C.int64_t)(unsafe.Pointer(&free)), (*C.int64_t)(unsafe.Pointer(&total)))
	return free, total
}

type CudaTensor struct {
	backend   *CudaBackend
	buf       C.CudaBufferRef
	rows      int
	cols      int
	sizeBytes int
	dtype     DataType
}

func (t *CudaTensor) Dims() (int, int) {
	return t.rows, t.cols
}

func (t *CudaTensor) DataType() DataType {
	if t.rows > 0 || t.cols > 0 { // If tensor was initialized
		if t.dtype == Float16 || t.dtype == Float32 || t.dtype == Float64 ||
			t.dtype == Int8 || t.dtype == Int16 || t.dtype == Int32 || t.dtype == Int64 ||
			t.dtype == Int || t.dtype == Uint8 || t.dtype == Uint16 || t.dtype == Uint32 ||
			t.dtype == Uint64 || t.dtype == Uint || t.dtype == Complex64 || t.dtype == Complex128 {
			return t.dtype // Explicitly set dtype
		}
	}
	if t.backend != nil && t.backend.useFP16 {
		return Float16
	}
	return Float32
}

func (t *CudaTensor) At(i, j int) float32 {
	// Slow path for debugging
	var val float32
	C.Cuda_CopyToHost(t.buf, C.int((i*t.cols+j)*4), unsafe.Pointer(&val), 4)
	return val
}

func (t *CudaTensor) Set(i, j int, v float32) {
	dtype := t.DataType()
	if dtype == Float16 {
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
	dtype := t.DataType()

	if dtype == Float16 {
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
	dtype := t.DataType()
	if dtype == Float16 {
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
	C.Cuda_CopyDeviceToDevice(t.buf, ft.buf, C.int(t.sizeBytes))
}

func (t *CudaTensor) Slice(i, k, j, l int) Tensor {
	r := k - i
	c := l - j
	out := t.backend.GetTensor(r, c)
	C.Cuda_Slice(t.backend.ctx, t.buf, out.(*CudaTensor).buf,
		C.int(i), C.int(j), C.int(r), C.int(c), C.int(t.cols))
	return out
}

func (t *CudaTensor) Paste(dstRow, dstCol int, src Tensor, srcRow, srcCol, rows, cols int) {
	st := src.(*CudaTensor)
	C.Cuda_Paste(t.backend.ctx, t.buf, st.buf,
		C.int(dstRow), C.int(dstCol), C.int(srcRow), C.int(srcCol),
		C.int(rows), C.int(cols), C.int(t.cols), C.int(st.cols))
}

func (t *CudaTensor) T() Tensor {
	panic("Transpose view not implemented for CUDA (handled in MatMul)")
}

func (t *CudaTensor) Mul(a, b Tensor) {
	t.Linear(a, b, nil)
}

func (t *CudaTensor) Add(other Tensor) {
	ot := other.(*CudaTensor)
	size := t.rows * t.cols
	C.Cuda_Add(t.backend.ctx, t.buf, ot.buf, t.buf, C.int(size))
}

func (t *CudaTensor) AddScalar(val float32) {
	size := t.rows * t.cols
	C.Cuda_AddScalar(t.backend.ctx, t.buf, C.float(val), t.buf, C.int(size))
}

func (t *CudaTensor) Scale(val float32) {
	size := t.rows * t.cols
	C.Cuda_Scale(t.backend.ctx, t.buf, C.float(val), t.buf, C.int(size))
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
	size := t.rows * t.cols
	C.Cuda_Tanh(t.backend.ctx, t.buf, t.buf, C.int(size))
}

func (t *CudaTensor) LayerNorm(gamma, beta Tensor, eps float32) {
	gg := gamma.(*CudaTensor)
	bb := beta.(*CudaTensor)
	C.Cuda_LayerNorm(t.backend.ctx, t.buf, gg.buf, bb.buf, t.buf, C.int(t.rows), C.int(t.cols), C.float(eps))
}

func (t *CudaTensor) AddLayerNorm(residual, gamma, beta Tensor, eps float32) {
	rt := residual.(*CudaTensor)
	gg := gamma.(*CudaTensor)
	bb := beta.(*CudaTensor)
	C.Cuda_AddLayerNorm(t.backend.ctx, rt.buf, gg.buf, bb.buf, t.buf, C.int(t.rows), C.int(t.cols), C.float(eps))
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

func (t *CudaTensor) Attention(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor {
	qt := q.(*CudaTensor)
	kt := k.(*CudaTensor)
	vt := v.(*CudaTensor)
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

func (t *CudaTensor) AttentionVarLen(q, k, v Tensor, lengths []int, numHeads int, scale float32) Tensor {
	panic("AttentionVarLen not implemented for CUDA")
}

func (t *CudaTensor) ApplyRoPE(batchSize, seqLen, numHeads, headDim int) {
	C.Cuda_ApplyRoPE(t.backend.ctx, t.buf, C.int(batchSize), C.int(seqLen), C.int(numHeads), C.int(headDim))
}

func (t *CudaTensor) ExtractToFlat(destination []float32, startOffset int) {
	data := t.ToHost()
	copy(destination[startOffset:], data)
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

func (t *CudaTensor) ExtractBytes() []byte {
	// For CUDA, since we don't have pinned memory logic fully wired for zero-copy to arrow yet,
	// we fall back to copy-to-host and then unsafe cast to bytes.
	// This ensures correctness but not yet full performance on Linux.

	// If FP16, ToHost returns float32 (converted).
	// If we want raw bytes of the *device* tensor (which might be FP16), we need a raw read.

	size := t.rows * t.cols
	if t.backend.useFP16 {
		// FP16 on device. We want those bytes if we are doing zero-copy transfer.
		// ToHost converts to FP32. We don't want that for "ExtractBytes" if the intention is to get raw transport format.
		// However, the interface contract implies "raw underlying byte representation".
		// If the tensor is FP16, we should return FP16 bytes.

		sizeBytes := size * 2
		out := make([]byte, sizeBytes)
		// Access C buffer directly
		C.Cuda_CopyToHost(t.buf, 0, unsafe.Pointer(&out[0]), C.int(sizeBytes))
		return out
	} else {
		// FP32
		sizeBytes := size * 4
		out := make([]byte, sizeBytes)
		C.Cuda_CopyToHost(t.buf, 0, unsafe.Pointer(&out[0]), C.int(sizeBytes))
		return out
	}
}

func (t *CudaTensor) HasNaN() (bool, error) {
	data := t.ToHost()
	for _, v := range data {
		if v != v {
			return true, nil
		}
	}
	return false, nil
}

func (t *CudaTensor) Cast(dtype DataType) Tensor {
	size := t.rows * t.cols
	currentDtype := t.DataType()

	if dtype == Float32 && currentDtype == Float16 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_F16_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float16 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float16).(*CudaTensor)
		C.Cuda_Cast_F32_to_F16(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float64 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float64).(*CudaTensor)
		C.Cuda_Cast_F32_to_F64(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Float64 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_F64_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Int32 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Int32).(*CudaTensor)
		C.Cuda_Cast_F32_to_I32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Int32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_I32_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Int64 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Int64).(*CudaTensor)
		C.Cuda_Cast_F32_to_I64(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Int64 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_I64_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Uint32 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Uint32).(*CudaTensor)
		C.Cuda_Cast_F32_to_U32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Uint32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_U32_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Uint64 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Uint64).(*CudaTensor)
		C.Cuda_Cast_F32_to_U64(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Uint64 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_U64_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Int8 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Int8).(*CudaTensor)
		C.Cuda_Cast_F32_to_I8(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Int8 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_I8_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Uint8 && currentDtype == Float32 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Uint8).(*CudaTensor)
		C.Cuda_Cast_F32_to_U8(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == Float32 && currentDtype == Uint8 {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, Float32).(*CudaTensor)
		C.Cuda_Cast_U8_to_F32(t.backend.ctx, t.buf, nt.buf, C.int(size))
		t.backend.Synchronize()
		return nt
	} else if dtype == currentDtype {
		nt := t.backend.GetTensorOfType(t.rows, t.cols, dtype).(*CudaTensor)
		nt.Copy(t)
		return nt
	}
	panic("Cast: Unsupported conversion from " + currentDtype.String() + " to " + dtype.String())
}
