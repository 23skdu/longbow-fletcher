//go:build darwin && metal

package device

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders
#cgo CFLAGS: -fobjc-arc
#include "metal_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	_ "embed"
	"fmt"
	"runtime"
	"unsafe"
)

//go:embed kernels.metal
var kernelsSource string

// Check interface compliance
var _ Backend = (*MetalBackend)(nil)
var _ Tensor = (*MetalTensor)(nil)

// bufferPoolEntry represents a pooled GPU buffer
type bufferPoolEntry struct {
	buf  C.MetalBufferRef
	size int // in bytes
}

type MetalBackend struct {
	ctx    C.MetalContextRef
	pool   []bufferPoolEntry // Simple pool of GPU buffers
	useFP16 bool             // Use FP16 precision for 2x GPU performance
}

func NewMetalBackend() *MetalBackend {
	cSrc := C.CString(kernelsSource)
	defer C.free(unsafe.Pointer(cSrc))
	
	ctx := C.Metal_Init(cSrc)
	if ctx == nil {
		panic("Failed to initialize Metal backend")
	}
	
	return &MetalBackend{ctx: ctx, pool: make([]bufferPoolEntry, 0, 64), useFP16: false}
}

// NewMetalBackendFP16 creates a Metal backend using FP16 precision for 2x GPU performance
func NewMetalBackendFP16() *MetalBackend {
	cSrc := C.CString(kernelsSource)
	defer C.free(unsafe.Pointer(cSrc))
	
	ctx := C.Metal_Init(cSrc)
	if ctx == nil {
		panic("Failed to initialize Metal backend")
	}
	
	return &MetalBackend{ctx: ctx, pool: make([]bufferPoolEntry, 0, 64), useFP16: true}
}

func (b *MetalBackend) Name() string {
	if b.useFP16 {
		return "Metal-FP16"
	}
	return "Metal"
}

func (b *MetalBackend) NewTensor(r, c int, data []float64) Tensor {
	size := r * c // number of elements
	
	var sizeBytes int
	var buf C.MetalBufferRef
	
	if b.useFP16 {
		// FP16: 2 bytes per element
		sizeBytes = size * 2
		buf = b.getPooledBuffer(sizeBytes)
		if buf == nil {
			buf = C.Metal_Alloc(C.MetalContextRef(b.ctx), C.int(sizeBytes))
		}
		
		if len(data) > 0 {
			// Convert []float64 to []uint16 (FP16 encoded)
			f16 := make([]uint16, size)
			for i, v := range data {
				f16[i] = float32ToFloat16(float32(v))
			}
			C.Metal_CopyToDevice(buf, 0, unsafe.Pointer(&f16[0]), C.int(sizeBytes))
		} else {
			C.Metal_Memset(buf, 0, 0, C.int(sizeBytes))
		}
	} else {
		// FP32: 4 bytes per element
		sizeBytes = size * 4
		buf = b.getPooledBuffer(sizeBytes)
		if buf == nil {
			buf = C.Metal_Alloc(C.MetalContextRef(b.ctx), C.int(sizeBytes))
		}
		
		if len(data) > 0 {
			f32 := make([]float32, size)
			for i, v := range data {
				f32[i] = float32(v)
			}
			C.Metal_CopyToDevice(buf, 0, unsafe.Pointer(&f32[0]), C.int(sizeBytes))
		} else {
			C.Metal_Memset(buf, 0, 0, C.int(sizeBytes))
		}
	}

	t := &MetalTensor{
		backend:  b,
		rows:     r,
		cols:     c,
		buf:      buf,
		offset:   0,
		sizeBytes: sizeBytes,
		ownsBuffer: true,
	}
	
	runtime.SetFinalizer(t, func(mt *MetalTensor) {
		if mt.ownsBuffer && mt.offset == 0 {
			mt.backend.returnToPool(mt.buf, mt.sizeBytes)
		}
	})
	
	return t
}

// FP16 conversion helpers
func float32ToFloat16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f))
	sign := (bits >> 31) & 1
	exp := (bits >> 23) & 0xFF
	frac := bits & 0x7FFFFF
	
	if exp == 255 { // Inf/NaN
		return uint16((sign << 15) | (0x1F << 10) | (frac >> 13))
	}
	if exp == 0 { // Zero/Denorm
		return uint16(sign << 15)
	}
	
	newExp := int(exp) - 127 + 15
	if newExp >= 31 { // Overflow -> Inf
		return uint16((sign << 15) | (0x1F << 10))
	}
	if newExp <= 0 { // Underflow -> Zero
		return uint16(sign << 15)
	}
	
	return uint16((sign << 15) | (uint32(newExp) << 10) | (frac >> 13))
}

func float16ToFloat32(h uint16) float32 {
	sign := (uint32(h) >> 15) & 1
	exp := (uint32(h) >> 10) & 0x1F
	frac := uint32(h) & 0x3FF
	
	if exp == 0 { // Zero/Denorm
		return 0.0
	}
	if exp == 31 { // Inf/NaN
		bits := (sign << 31) | (0xFF << 23) | (frac << 13)
		return *(*float32)(unsafe.Pointer(&bits))
	}
	
	newExp := exp - 15 + 127
	bits := (sign << 31) | (newExp << 23) | (frac << 13)
	return *(*float32)(unsafe.Pointer(&bits))
}

func (b *MetalBackend) getPooledBuffer(sizeBytes int) C.MetalBufferRef {
	// Find a buffer of exact size or slightly larger (up to 2x)
	for i, entry := range b.pool {
		if entry.size >= sizeBytes && entry.size <= sizeBytes*2 {
			// Remove from pool and return
			b.pool = append(b.pool[:i], b.pool[i+1:]...)
			return entry.buf
		}
	}
	return nil
}

func (b *MetalBackend) returnToPool(buf C.MetalBufferRef, sizeBytes int) {
	// Limit pool size
	if len(b.pool) >= 128 {
		// Pool full, free the oldest entry
		C.Metal_FreeBuffer(b.ctx, b.pool[0].buf)
		b.pool = b.pool[1:]
	}
	b.pool = append(b.pool, bufferPoolEntry{buf: buf, size: sizeBytes})
}

func (b *MetalBackend) GetTensor(r, c int) Tensor {
	return b.NewTensor(r, c, nil)
}

func (b *MetalBackend) PutTensor(t Tensor) {
	// Tensor will be returned to pool via finalizer
	// Optionally we could force return here if needed
}

func (b *MetalBackend) Synchronize() {
	C.Metal_Synchronize(b.ctx)
}

type MetalTensor struct {
	backend    *MetalBackend
	buf        C.MetalBufferRef
	rows       int
	cols       int
	trans      bool // Transposed view flag
	offset     int  // in bytes, offset into the underlying buffer
	sizeBytes  int  // total size of owned buffer in bytes
	ownsBuffer bool // true if this tensor owns the buffer (for pooling)
}

func (t *MetalTensor) Dims() (int, int) {
	if t.trans {
		return t.cols, t.rows
	}
	return t.rows, t.cols
}

func (t *MetalTensor) At(i, j int) float64 {
	// Very slow!
	// Copy to host
	
	// Check bounds
	rows, cols := t.Dims()
	if i < 0 || i >= rows || j < 0 || j >= cols {
		panic("Index out of bounds")
	}

	// For prototype, just use ToHost which handles logic?
	// No, ToHost copies raw buffer for now.
	raw := t.rawHostCopy()
	// Raw is typically row-major of physical buffer
	// Physical buffer is t.rows x t.cols
	
	var val float32
	if t.trans {
		// virtual (i, j) -> physical (j, i)
		val = raw[j*t.cols + i] // j is phys row, i is phys col
	} else {
		val = raw[i*t.cols + j]
	}
	return float64(val)
}

func (t *MetalTensor) Set(i, j int, v float64) {
	var idx int
	if t.trans {
		idx = j*t.cols + i 
	} else {
		idx = i*t.cols + j
	}
	
	if t.backend.useFP16 {
		// FP16: 2 bytes per element - use conversion and direct memory write
		f16Val := float32ToFloat16(float32(v))
		byteOffset := t.offset + idx*2
		// Write FP16 value via Metal_CopyToDevice
		C.Metal_CopyToDevice(t.buf, C.int(byteOffset), unsafe.Pointer(&f16Val), 2)
	} else {
		// FP32: 4 bytes per element
		C.Metal_SetAt(t.buf, C.int(t.offset + idx*4), C.float(v))
	}
}

func (t *MetalTensor) rawHostCopy() []float32 {
	t.backend.Synchronize()
	
	size := t.rows * t.cols
	
	if t.backend.useFP16 {
		// Read FP16 data and convert to float32
		raw16 := make([]uint16, size)
		C.Metal_CopyToHost(t.buf, C.int(t.offset), unsafe.Pointer(&raw16[0]), C.int(size*2))
		
		raw := make([]float32, size)
		for i, h := range raw16 {
			raw[i] = float16ToFloat32(h)
		}
		return raw
	}
	
	// FP32 path
	raw := make([]float32, size)
	C.Metal_CopyToHost(t.buf, C.int(t.offset), unsafe.Pointer(&raw[0]), C.int(size*4))
	return raw
}

// ToHost copies data back to CPU.
func (t *MetalTensor) ToHost() []float64 {
	// rawHostCopy gives us the physical buffer contents corresponding to this tensor view.
	// If t.trans, we need to handle it.
	
	raw := t.rawHostCopy()
	
	data := make([]float64, len(raw))
	for i, v := range raw {
		data[i] = float64(v)
	}
	
	// If transposed, we must shuffle.
	if t.trans {
		rows, cols := t.cols, t.rows // logical dimensions
		out := make([]float64, len(raw))
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				// Logical (i,j) maps to physical (j,i)
				// Physical buffer is t.rows x t.cols
				// So, raw[j*t.cols + i] is the element at physical (j,i)
				out[i*cols+j] = data[j*t.cols+i]
			}
		}
		return out
	}
	
	return data
}

func (t *MetalTensor) Data() []float64 {
	// Return nil to indicate data is on device
	return nil
}

// CopyFromFloat64 copies []float64 data to GPU in a single bulk operation.
// This is much faster than using Set() for each element.
func (t *MetalTensor) CopyFromFloat64(data []float64) {
	size := t.rows * t.cols
	if len(data) != size {
		panic("CopyFromFloat64: size mismatch")
	}
	
	if t.backend.useFP16 {
		// Batch convert to FP16 and upload
		f16 := make([]uint16, size)
		for i, v := range data {
			f16[i] = float32ToFloat16(float32(v))
		}
		C.Metal_CopyToDevice(t.buf, C.int(t.offset), unsafe.Pointer(&f16[0]), C.int(size*2))
	} else {
		// FP32 path
		f32 := make([]float32, size)
		for i, v := range data {
			f32[i] = float32(v)
		}
		C.Metal_CopyToDevice(t.buf, C.int(t.offset), unsafe.Pointer(&f32[0]), C.int(size*4))
	}
}

func (t *MetalTensor) Copy(from Tensor) {
	_, ok := from.(*MetalTensor)
	if !ok {
		panic("Cross-backend copy not supported")
	}
	// TODO: Metal_CopyBuffer
}

func (t *MetalTensor) Slice(i, k, j, l int) Tensor {
	// Validate bounds
	if i < 0 || k > t.rows || j < 0 || l > t.cols || i >= k || j >= l {
		panic("Invalid slice bounds")
	}
	
	// Metal handles 1D buffers. 2D slicing is tricky unless strides are supported or full-width slice.
	// If full width slice (j=0, l=cols), it's a contiguous chunk.
	if j == 0 && l == t.cols {
		// Contiguous rows
		newRows := k - i
		newCols := t.cols
		
		// Byte offset depends on precision
		var rowBytes int
		if t.backend.useFP16 {
			rowBytes = t.cols * 2 // FP16: 2 bytes
		} else {
			rowBytes = t.cols * 4 // FP32: 4 bytes
		}
		
		byteOffset := t.offset + i * rowBytes
		
		return &MetalTensor{
			backend: t.backend,
			rows:    newRows,
			cols:    newCols,
			trans:   t.trans,
			buf:     t.buf,
			offset:  byteOffset,
		}
	}
	
	panic("Non-row slice not supported in Metal yet")
}

func (t *MetalTensor) T() Tensor {
	return &MetalTensor{
		backend: t.backend,
		buf:     t.buf,
		rows:    t.rows,
		cols:    t.cols,
		trans:   !t.trans,
		offset:  t.offset, // Transpose is a view, offset remains the same
	}
}

func (t *MetalTensor) Mul(a, b Tensor) {
	ma, ok1 := a.(*MetalTensor)
	mb, ok2 := b.(*MetalTensor)
	if !ok1 || !ok2 {
		panic("Mixed backend Mul")
	}
	
	r, common1 := ma.Dims()
	common2, c := mb.Dims()
	
	if common1 != common2 {
		panic(fmt.Sprintf("Dimension mismatch: %dx%d * %dx%d", r, common1, common2, c))
	}
	
	if t.backend.useFP16 {
		// FP16 MatMul for 2x performance
		C.Metal_MatMul_F16(t.backend.ctx, 
			ma.buf, C.int(ma.offset), C.bool(ma.trans),
			mb.buf, C.int(mb.offset), C.bool(mb.trans),
			t.buf, C.int(t.offset),
			C.int(r), C.int(c), C.int(common1))
	} else {
		C.Metal_MatMul(t.backend.ctx, 
			ma.buf, C.int(ma.offset), C.bool(ma.trans),
			mb.buf, C.int(mb.offset), C.bool(mb.trans),
			t.buf, C.int(t.offset),
			C.int(r), C.int(c), C.int(common1))
	}
}

func (t *MetalTensor) Add(other Tensor) {
	ot, ok := other.(*MetalTensor)
	if !ok { panic("Mixed backend Add") }
	
	size := t.rows * t.cols
	C.Metal_Add(t.backend.ctx, t.buf, C.int(t.offset), ot.buf, C.int(ot.offset), t.buf, C.int(t.offset), C.int(size))
}

func (t *MetalTensor) AddScalar(val float64) {
	size := t.rows * t.cols
	C.Metal_AddScalar(t.backend.ctx, t.buf, C.int(t.offset), C.float(val), t.buf, C.int(t.offset), C.int(size))
}

func (t *MetalTensor) Scale(val float64) {
	size := t.rows * t.cols
	if t.backend.useFP16 {
		f16Val := float32ToFloat16(float32(val))
		C.Metal_Scale_F16(t.backend.ctx, t.buf, C.int(t.offset), C.uint16_t(f16Val), t.buf, C.int(t.offset), C.int(size))
	} else {
		C.Metal_Scale(t.backend.ctx, t.buf, C.int(t.offset), C.float(val), t.buf, C.int(t.offset), C.int(size))
	}
}

func (t *MetalTensor) AddBias(bias []float64) {
	// Upload bias to temp buffer
	// Or cache bias?
	// For now temp buffer.
	// Optimization: Keep bias on device.
	// But signature takes []float64.
	// BertModel loads bias as []float64 usually? 
	// Wait, BertModel stores biases as []float64 in some structs (e.g. QueryBias), but as Tensor in others?
	// QueryBias is []float64.
	
	biasSize := len(bias) * 4
	biasBuf := C.Metal_Alloc(t.backend.ctx, C.int(biasSize))
	defer C.Metal_FreeBuffer(t.backend.ctx, biasBuf)
	
	f32 := make([]float32, len(bias))
	for i, v := range bias {
		f32[i] = float32(v)
	}
	C.Metal_CopyToDevice(biasBuf, 0, unsafe.Pointer(&f32[0]), C.int(biasSize))
	
	// Check dim
	_, cols := t.Dims()
	if cols != len(bias) {
		panic("AddBias dim mismatch")
	}
	
	C.Metal_AddBias(t.backend.ctx, t.buf, C.int(t.offset), biasBuf, 0, C.int(t.rows), C.int(cols))
}

func (t *MetalTensor) Softmax() {
	// In-Place Softmax. Input=t.buf, Output=t.buf
	// Rows = t.rows (or t.cols if transposed?)
	// Softmax is applied to LAST dim usually in BERT (per row of embeddings).
	// If transposed, last dim is t.rows?
	// Dims() returns (r, c). Softmax over c.
	// if t.trans: Dims -> (cols, rows). "rows" are columns. Softmax over dim 1 (rows).
	// But underlying buffer is rows x cols. 
	// If trans, we are viewing it as cols x rows.
	// Softmax should reduce along dim 1.
	
	// If trans, memory is col-major logic effectively.
	// Row i in view = Col i in buffer (strided).
	// Our kernel assumes contiguous rows.
	// Softmax on stride is hard.
	// Panic if softmax on transposed tensor?
	// BERT does not do Softmax on transposed tensor.
	// BertSelfAttention: attentionScores = Q * K^T. then Softmax(attentionScores).
	// attentionScores is NOT transposed.
	
	if t.trans {
		panic("Softmax on transposed tensor not supported yet")
	}
	
	if t.backend.useFP16 {
		C.Metal_Softmax_F16(t.backend.ctx, t.buf, C.int(t.offset), t.buf, C.int(t.offset), C.int(t.rows), C.int(t.cols))
	} else {
		C.Metal_Softmax(t.backend.ctx, t.buf, C.int(t.offset), t.buf, C.int(t.offset), C.int(t.rows), C.int(t.cols))
	}
}

func (t *MetalTensor) Gather(indices []int) Tensor {
	// 1. Upload indices to GPU
	indicesSize := len(indices) * 4 // int32
	indicesBuf := C.Metal_Alloc(t.backend.ctx, C.int(indicesSize))
	defer C.Metal_FreeBuffer(t.backend.ctx, indicesBuf)
	
	i32 := make([]int32, len(indices))
	for i, v := range indices {
		i32[i] = int32(v)
	}
	C.Metal_CopyToDevice(indicesBuf, 0, unsafe.Pointer(&i32[0]), C.int(indicesSize))
	
	// 2. Allocate Output
	// Output dims: len(indices) x cols
	// Use t.cols (physical). Dims() might returned swapped if transposed.
	// Gather on transposed?
	if t.trans {
		panic("Gather on transposed tensor not supported")
	}
	
	outRows := len(indices)
	outCols := t.cols 
	
	output := t.backend.NewTensor(outRows, outCols, nil)
	mtOut := output.(*MetalTensor)
	
	// 3. Dispatch
	C.Metal_Gather(t.backend.ctx, t.buf, C.int(t.offset), indicesBuf, 0, mtOut.buf, C.int(mtOut.offset), C.int(outRows), C.int(outCols))
	
	return output
}

func (t *MetalTensor) Gelu() {
	size := t.rows * t.cols
	C.Metal_Gelu(t.backend.ctx, t.buf, C.int(t.offset), t.buf, C.int(t.offset), C.int(size))
}

func (t *MetalTensor) Tanh() {
	size := t.rows * t.cols
	C.Metal_Tanh(t.backend.ctx, t.buf, C.int(t.offset), t.buf, C.int(t.offset), C.int(size))
}

func (t *MetalTensor) LayerNorm(gamma, beta Tensor, eps float64) {
	gt, ok1 := gamma.(*MetalTensor)
	bt, ok2 := beta.(*MetalTensor)
	if !ok1 || !ok2 { panic("Mixed backend LayerNorm") }
	
	C.Metal_LayerNorm(t.backend.ctx, 
		t.buf, C.int(t.offset), gt.buf, C.int(gt.offset), bt.buf, C.int(bt.offset), t.buf, C.int(t.offset),
		C.int(t.rows), C.int(t.cols), C.float(eps))
}
