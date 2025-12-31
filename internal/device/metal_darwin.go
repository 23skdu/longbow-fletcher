//go:build darwin && metal

package device

/*
#cgo LDFLAGS: -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate
#cgo CFLAGS: -fobjc-arc
#include "metal_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	_ "embed"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

//go:embed kernels_darwin.metal
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
	ctx            C.MetalContextRef
	mu             sync.Mutex
	buckets        map[int][]bufferPoolEntry // Safe to reuse
	pendingBuckets map[int][]bufferPoolEntry // In flight on GPU
	useFP16        bool                      // Use FP16 precision
}

func getBucket(size int) int {
	if size <= 0 {
		return 0
	}
	// Use log2 to determine bucket. E.g., 1-2 bytes -> bucket 1, 3-4 bytes -> bucket 2, 5-8 bytes -> bucket 3, etc.
	return int(math.Ceil(math.Log2(float64(size))))
}

func NewMetalBackend() *MetalBackend {
	cSrc := C.CString(kernelsSource)
	defer C.free(unsafe.Pointer(cSrc))
	
	ctx := C.Metal_Init(cSrc)
	if ctx == nil {
		panic("Failed to initialize Metal backend")
	}
	
	return &MetalBackend{
		ctx:            ctx,
		buckets:        make(map[int][]bufferPoolEntry),
		pendingBuckets: make(map[int][]bufferPoolEntry),
		useFP16:        false,
	}
}

// NewMetalBackendFP16 creates a Metal backend using FP16 precision for 2x GPU performance
func NewMetalBackendFP16() *MetalBackend {
	cSrc := C.CString(kernelsSource)
	defer C.free(unsafe.Pointer(cSrc))
	
	ctx := C.Metal_Init(cSrc)
	if ctx == nil {
		panic("Failed to initialize Metal backend")
	}
	
	return &MetalBackend{
		ctx:            ctx,
		buckets:        make(map[int][]bufferPoolEntry),
		pendingBuckets: make(map[int][]bufferPoolEntry),
		useFP16:        true,
	}
}

func (b *MetalBackend) Name() string {
	if b.useFP16 {
		return "Metal-FP16"
	}
	return "Metal"
}

func (b *MetalBackend) NewTensor(r, c int, data []float32) Tensor {
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
			// Convert []float32 to []uint16 (FP16 encoded)
			f16 := make([]uint16, size)
			for i, v := range data {
				f16[i] = Float32ToFloat16(v)
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
			// Direct copy for FP32
			C.Metal_CopyToDevice(buf, 0, unsafe.Pointer(&data[0]), C.int(sizeBytes))
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
		dtype: func() DataType { if b.useFP16 { return Float16 }; return Float32 }(),
	}
	
	runtime.SetFinalizer(t, func(mt *MetalTensor) {
		if mt.ownsBuffer && mt.offset == 0 {
			mt.backend.returnToPool(mt.buf, mt.sizeBytes)
		}
	})
	
	return t
}

	// FP16 conversion helpers are now in utils.go

func (b *MetalBackend) getPooledBuffer(sizeBytes int) C.MetalBufferRef {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Drain pending buffers if GPU is done
	if bool(C.Metal_IsCompleted(b.ctx)) {
		for bucket, entries := range b.pendingBuckets {
			b.buckets[bucket] = append(b.buckets[bucket], entries...)
		}
		b.pendingBuckets = make(map[int][]bufferPoolEntry)
	}

	bucket := getBucket(sizeBytes)
	for i := bucket; i <= bucket+2; i++ {
		list := b.buckets[i]
		if len(list) > 0 {
			bestIdx := -1
			for idx, entry := range list {
				if entry.size >= sizeBytes {
					if bestIdx == -1 || entry.size < list[bestIdx].size {
						bestIdx = idx
					}
				}
			}
			if bestIdx != -1 {
				buf := list[bestIdx].buf
				b.buckets[i] = append(list[:bestIdx], list[bestIdx+1:]...)
				
				// Metrics: Hit
				poolHits.Inc()
				poolSizeBytes.Sub(float64(list[bestIdx].size))
				poolBuffers.Dec()
				
				return buf
			}
		}
	}
	
	// Metrics: Miss
	poolMisses.Inc()
	return nil
}

// HasNaN checks if the tensor contains any NaN values.
// This is a blocking operation that synchronizes with the GPU.
func (t *MetalTensor) HasNaN() (bool, error) {
	// Allocate a 4-byte result buffer on GPU (int32)
	// We use an existing pool or just alloc/free for now since checks are rare/sparse?
	// Or we can use a small persistent buffer?
	// Let's alloc new for safety/simplicity first.
	// 4 bytes = 1 x int32
	
	resBuf := C.Metal_Alloc(t.backend.ctx, 4)
	if resBuf == nil {
		return false, fmt.Errorf("failed to allocate result buffer for NaN check")
	}
	defer C.Metal_FreeBuffer(t.backend.ctx, resBuf)
	
	// Initialize to 0
	C.Metal_Memset(resBuf, 0, 0, 4) // Memset 0
	
	count := t.rows * t.cols
	
	if t.backend.useFP16 {
		C.Metal_CheckNaN_F16(t.backend.ctx, t.buf, C.int(t.offset), C.int(count), resBuf)
	} else {
		C.Metal_CheckNaN_F32(t.backend.ctx, t.buf, C.int(t.offset), C.int(count), resBuf)
	}
	
	// Blocking read
	t.backend.Synchronize()
	
	var result int32
	C.Metal_ExtractBytes(resBuf, 0, unsafe.Pointer(&result), 4)
	
	return result != 0, nil
}

func (b *MetalBackend) returnToPool(buf C.MetalBufferRef, sizeBytes int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	bucket := getBucket(sizeBytes)
	// Add to pending (in-flight) queue until next getPooledBuffer drains it
	b.pendingBuckets[bucket] = append(b.pendingBuckets[bucket], bufferPoolEntry{buf: buf, size: sizeBytes})
	
	// Metrics: Return
	poolSizeBytes.Add(float64(sizeBytes))
	poolBuffers.Inc()
}

func (b *MetalBackend) GetTensor(r, c int) Tensor {
	return b.NewTensor(r, c, nil)
}

func (b *MetalBackend) PutTensor(t Tensor) {
	mt, ok := t.(*MetalTensor)
	if !ok || mt.buf == nil {
		return
	}

	// Only return if we own it and it's not a slice
	if mt.ownsBuffer && mt.offset == 0 {
		b.returnToPool(mt.buf, mt.sizeBytes)
		mt.buf = nil // Prevent double-return
		mt.ownsBuffer = false
		runtime.SetFinalizer(mt, nil)
	}
}

func (b *MetalBackend) Synchronize() {
	C.Metal_Synchronize(b.ctx)
}

func (b *MetalBackend) DeviceCount() int {
	return 1 // Single Metal device for now
}

func (b *MetalBackend) SetDevice(index int) {
	if index != 0 {
		panic("Invalid Metal device index")
	}
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
	dtype      DataType
}

func (t *MetalTensor) Dims() (int, int) {
	if t.trans {
		return t.cols, t.rows
	}
	return t.rows, t.cols
}

func (t *MetalTensor) At(i, j int) float32 {
	// Very slow! CPU readback
	
	// Check bounds
	rows, cols := t.Dims()
	if i < 0 || i >= rows || j < 0 || j >= cols {
		panic("Index out of bounds")
	}

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
	return val
}

func (t *MetalTensor) Set(i, j int, v float32) {
	var idx int
	if t.trans {
		idx = j*t.cols + i 
	} else {
		idx = i*t.cols + j
	}
	
	if t.backend.useFP16 {
		// FP16: 2 bytes per element - use conversion and direct memory write
		f16Val := Float32ToFloat16(v)
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
			raw[i] = Float16ToFloat32(h)
		}
		return raw
	}
	
	// FP32 path
	raw := make([]float32, size)
	C.Metal_CopyToHost(t.buf, C.int(t.offset), unsafe.Pointer(&raw[0]), C.int(size*4))
	return raw
}

func (t *MetalTensor) Data() []float32 {
	t.backend.Synchronize()
	
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		return nil
	}
	
	size := t.rows * t.cols
	if t.backend.useFP16 {
		// FP16 data. We still need to convert it to FP32 for the Go layers, 
		// but we can do it more efficiently than ToHost().
		// For now, ToHost is safer if it's FP16. 
		// But let's provide a way to get raw host access if it were FP32.
		return nil // Force ToHost for FP16 for now to handle conversion
	}
	
	// FP32 path: direct slice header trick
	// We must use the offset.
	dataPtr := unsafe.Pointer(uintptr(ptr) + uintptr(t.offset))
	return (*[1 << 30]float32)(dataPtr)[:size:size]
}

// ToHost copies data back to CPU features.
func (t *MetalTensor) ToHost() []float32 {
	// If it's FP32 and not transposed, we can use zero-copy then copy.
	if !t.backend.useFP16 && !t.trans {
		d := t.Data()
		if d != nil {
			out := make([]float32, len(d))
			copy(out, d)
			return out
		}
	}
	
	// Otherwise (FP16 or transposed), use the robust rawHostCopy
	raw := t.rawHostCopy()
	
	if t.trans {
		rows, cols := t.cols, t.rows // logical dimensions
		out := make([]float32, len(raw))
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				out[i*cols+j] = raw[j*t.cols+i]
			}
		}
		return out
	}
	
	return raw
}

// CopyFromFloat32 copies []float32 data to GPU in a single bulk operation.
// This is much faster than using Set() for each element.
func (t *MetalTensor) CopyFromFloat32(data []float32) {
	size := t.rows * t.cols
	if len(data) != size {
		panic("CopyFromFloat32: size mismatch")
	}
	
	if t.backend.useFP16 {
		// Batch convert to FP16 and upload
		f16 := make([]uint16, size)
		for i, v := range data {
			f16[i] = Float32ToFloat16(v)
		}
		C.Metal_CopyToDevice(t.buf, C.int(t.offset), unsafe.Pointer(&f16[0]), C.int(size*2))
	} else {
		// FP32 path
		C.Metal_CopyToDevice(t.buf, C.int(t.offset), unsafe.Pointer(&data[0]), C.int(size*4))
	}
}

func (t *MetalTensor) Copy(from Tensor) {
	ft, ok := from.(*MetalTensor)
	if !ok {
		panic("Cross-backend copy not supported")
	}
	
	// Check size
	size := t.rows * t.cols
	fSize := ft.rows * ft.cols
	if size != fSize {
		panic(fmt.Sprintf("Copy size mismatch: %d vs %d", size, fSize))
	}
	
	// We can treat as 1D copy (1 row, N cols) for simplicity with CopySubmatrix
	// provided strides align (which they do for 1D/contiguous 2D).
	
	// Determine correct kernel based on type.
	// If both usage FP16 vs FP32 matches?
	// Actually CopySubmatrix takes float pointers for FP32 version and half pointers for F16.
	// We should check backend type or tensor dtype.
	
	// If types differ, we need Cast, not Copy. Copy assumes same type.
	if t.dtype != ft.dtype {
		panic("Copy requires matching data types (use Cast)")
	}
	
	if t.dtype == Float16 {
		C.Metal_CopySubmatrix_F16(t.backend.ctx, ft.buf, C.int(ft.offset), C.int(size),
			t.buf, C.int(t.offset), C.int(size),
			0, 0, 1, C.int(size))
	} else {
		C.Metal_CopySubmatrix(t.backend.ctx, ft.buf, C.int(ft.offset), C.int(size),
			t.buf, C.int(t.offset), C.int(size),
			0, 0, 1, C.int(size))
	}
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
	} else {
		// General sub-tensor slicing (including column slicing)
		// Since MetalTensor is contiguous, we MUST create a COPY if we slice columns
		
		newRows := k - i
		newCols := l - j
		
		res := t.backend.NewTensor(newRows, newCols, nil)
		resT := res.(*MetalTensor)
		
		if t.backend.useFP16 {
			C.Metal_CopySubmatrix_F16(t.backend.ctx, t.buf, C.int(t.offset), C.int(t.cols),
				resT.buf, C.int(resT.offset), C.int(newCols),
				C.int(i), C.int(j), C.int(newRows), C.int(newCols))
		} else {
			C.Metal_CopySubmatrix(t.backend.ctx, t.buf, C.int(t.offset), C.int(t.cols),
				resT.buf, C.int(resT.offset), C.int(newCols),
				C.int(i), C.int(j), C.int(newRows), C.int(newCols))
		}
		
		return res
	}
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

func (t *MetalTensor) AddScalar(val float32) {
	size := t.rows * t.cols
	C.Metal_AddScalar(t.backend.ctx, t.buf, C.int(t.offset), C.float(val), t.buf, C.int(t.offset), C.int(size))
}

func (t *MetalTensor) Scale(val float32) {
	size := t.rows * t.cols
	if t.backend.useFP16 {
		f16Val := Float32ToFloat16(val)
		C.Metal_Scale_F16(t.backend.ctx, t.buf, C.int(t.offset), C.uint16_t(f16Val), t.buf, C.int(t.offset), C.int(size))
	} else {
		C.Metal_Scale(t.backend.ctx, t.buf, C.int(t.offset), C.float(val), t.buf, C.int(t.offset), C.int(size))
	}
}

func (t *MetalTensor) AddBias(bias Tensor) {
	// Cast bias to MetalTensor
	bt, ok := bias.(*MetalTensor)
	if !ok {
		panic("AddBias: bias must be MetalTensor")
	}
	
	// Verify dimensions (bias must match rows or cols depending on logic, effectively cols)
	// Bias dims should be equal to t.cols
	if bt.rows * bt.cols != t.cols {
		panic("AddBias: bias dimension mismatch")
	}
	
	if t.backend.useFP16 {
		// Use FP16 kernel with persistent bias tensor
		// Assume Metal_AddBias_F16 supports broadcasting or similar logic as Metal_AddBias
		C.Metal_AddBias_F16(t.backend.ctx, t.buf, C.int(t.offset), bt.buf, C.int(bt.offset), t.buf, C.int(t.offset), C.int(t.rows), C.int(t.cols))
	} else {
		// FP32 Path
		C.Metal_AddBias(t.backend.ctx, t.buf, C.int(t.offset), bt.buf, C.int(bt.offset), C.int(t.rows), C.int(t.cols))
	}
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

func (t *MetalTensor) LayerNorm(gamma, beta Tensor, eps float32) {
	gt, ok1 := gamma.(*MetalTensor)
	bt, ok2 := beta.(*MetalTensor)
	if !ok1 || !ok2 { panic("Mixed backend LayerNorm") }
	
	if t.backend.useFP16 {
		C.Metal_LayerNorm_F16(t.backend.ctx, 
			t.buf, C.int(t.offset), gt.buf, C.int(gt.offset), bt.buf, C.int(bt.offset), t.buf, C.int(t.offset),
			C.int(t.rows), C.int(t.cols), C.float(eps))
	} else {
		C.Metal_LayerNorm(t.backend.ctx, 
			t.buf, C.int(t.offset), gt.buf, C.int(gt.offset), bt.buf, C.int(bt.offset), t.buf, C.int(t.offset),
			C.int(t.rows), C.int(t.cols), C.float(eps))
	}
}

func (t *MetalTensor) Linear(input, weight, bias Tensor) Tensor {
	it := input.(*MetalTensor)
	wt := weight.(*MetalTensor)
	
	// Check dimensions
	r, ic := it.Dims()
	wic, oc := wt.Dims()
	
	if ic != wic {
		panic("Linear dimension mismatch")
	}
	
	// Create Result
	res := t.backend.NewTensor(r, oc, nil)
	rst := res.(*MetalTensor)
	
	if t.backend.useFP16 {
		// Use MPSGraph Fused Op
		var bt *MetalTensor
		if bias != nil {
			bt = bias.(*MetalTensor)
		} else {
			// Zero bias if nil? 
			// Graph expects bias. Create dummy zero bias?
			// For high perf, caller should ensure bias exists (BertModel does).
			// If nil, maybe just Mul?
			// Let's panic for now or handle gracefully.
			panic("Linear requires bias for now")
		}
		
		C.Metal_Linear_Graph(t.backend.ctx, 
			it.buf, C.int(it.offset), C.int(it.rows), C.int(it.cols),
			wt.buf, C.int(wt.offset), C.int(wt.cols),
			bt.buf, C.int(bt.offset),
			rst.buf, C.int(rst.offset))
			
		return res
	} else {
		// Fallback FP32
		res.Mul(input, weight)
		if bias != nil {
			res.AddBias(bias)
		}
		return res
	}
}

// override LinearActivation for SwiGLU if needed
func (t *MetalTensor) LinearActivation(input, weight, bias Tensor, activation ActivationType) Tensor {
	if activation == ActivationSwiGLU && t.backend.useFP16 {
		it := input.(*MetalTensor)
		wt := weight.(*MetalTensor)
		bt := bias.(*MetalTensor)
		r, _ := it.Dims()
		_, oc2 := wt.Dims()
		oc := oc2 / 2

		// 1. Linear projection to 2x intermediate size
		res2i := t.Linear(it, wt, bt).(*MetalTensor)
		
		// 2. SwiGLU reduction to 1x intermediate size
		res := t.backend.NewTensor(r, oc, nil)
		rst := res.(*MetalTensor)
		C.Metal_SwiGLU_F16(t.backend.ctx, res2i.buf, C.int(res2i.offset), rst.buf, C.int(rst.offset), C.int(r), C.int(oc))
		
		t.backend.PutTensor(res2i)
		return res
	}
	// Normal path or FP32 fallback
	return t.linearActivationInternal(input, weight, bias, activation)
}

func (t *MetalTensor) linearActivationInternal(input, weight, bias Tensor, activation ActivationType) Tensor {
	if t.backend.useFP16 {
		it := input.(*MetalTensor)
		wt := weight.(*MetalTensor)
		bt := bias.(*MetalTensor)
		r, _ := it.Dims()
		_, oc := wt.Dims()
		res := t.backend.NewTensor(r, oc, nil)
		rst := res.(*MetalTensor)
		
		C.Metal_LinearActivation_Graph(t.backend.ctx,
			it.buf, C.int(it.offset), C.int(it.rows), C.int(it.cols),
			wt.buf, C.int(wt.offset), C.int(wt.cols),
			bt.buf, C.int(bt.offset),
			rst.buf, C.int(rst.offset),
			C.int(activation))
			
		return res
	} else {
		res := t.Linear(input, weight, bias)
		switch activation {
		case ActivationGELU:
			res.Gelu()
		case ActivationTanh:
			res.Tanh()
		case ActivationSoftmax:
			res.Softmax()
		}
		return res
	}
}

func (t *MetalTensor) Attention(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor {
	if t.backend.useFP16 {
		return t.FlashAttention(q, k, v, batchSize, seqLen, numHeads, scale)
	}
	panic("Attention not implemented for Metal FP32")
}

func (t *MetalTensor) FlashAttention(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor {
	if !t.backend.useFP16 {
		panic("FlashAttention only supports FP16")
	}

	qt := q.(*MetalTensor)
	kt := k.(*MetalTensor)
	vt := v.(*MetalTensor)

	// Dimensions
	// q: [batch_size * seq_len, hidden_size]
	r, hiddenSize := qt.Dims()
	if r != batchSize*seqLen {
		panic(fmt.Sprintf("FlashAttention: dims mismatch q.rows=%d expected=%d", r, batchSize*seqLen))
	}
	
	headDim := hiddenSize / numHeads
	if headDim*numHeads != hiddenSize {
		panic("FlashAttention: hiddenSize not divisible by numHeads")
	}
	
	// Check constraints
	if headDim > 128 {
		// Fallback to Graph implementation if headDim is too large for our kernel
		return t.AttentionGraph(q, k, v, batchSize, seqLen, numHeads, scale)
	}

	// Result tensor
	result := t.backend.NewTensor(r, hiddenSize, nil)
	rst := result.(*MetalTensor)

	// Strides (Assuming packed row-major layout (Batch*Seq, Hidden))
	// row_stride = hiddenSize
	// head_stride = headDim
	// batch_stride = seqLen * hiddenSize
	
	rowStride := hiddenSize
	headStride := headDim
	batchStride := seqLen * hiddenSize
	
	totalBatches := batchSize * numHeads
	
	C.Metal_FlashAttention(t.backend.ctx,
		qt.buf, C.int(qt.offset),
		kt.buf, C.int(kt.offset),
		vt.buf, C.int(vt.offset),
		rst.buf, C.int(rst.offset),
		C.int(seqLen), C.int(headDim), C.float(scale),
		C.int(batchStride), C.int(headStride), C.int(rowStride),
		C.int(numHeads), C.int(totalBatches))
		
	return result
}

func (t *MetalTensor) AttentionGraph(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor {
	qt := q.(*MetalTensor)
	kt := k.(*MetalTensor)
	vt := v.(*MetalTensor)
	
	r, c := qt.Dims()
	result := t.backend.NewTensor(r, c, nil)
	rst := result.(*MetalTensor)
	
	C.Metal_Attention_Graph(t.backend.ctx,
		qt.buf, C.int(qt.offset),
		kt.buf, C.int(kt.offset),
		vt.buf, C.int(vt.offset),
		rst.buf, C.int(rst.offset),
		C.int(batchSize), C.int(seqLen), C.int(c), C.float(scale))
		
	return result
}

func (t *MetalTensor) ExtractTo(dest [][]float32, start int) {
	t.backend.Synchronize()
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		return
	}
	
	r, c := t.rows, t.cols
	if t.trans {
		// Fallback to ToHost for transposed
		data := t.ToHost()
		for i := 0; i < r; i++ {
			row := make([]float32, c)
			copy(row, data[i*c:(i+1)*c])
			dest[start+i] = row
		}
		return
	}

	numWorkers := 4
	if r < numWorkers {
		numWorkers = r
	}
	
	chunkSize := (r + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wStart := w * chunkSize
		if wStart >= r {
			break
		}
		wEnd := wStart + chunkSize
		if wEnd > r {
			wEnd = r
		}
		
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			if t.backend.useFP16 {
				for i := s; i < e; i++ {
					row := make([]float32, c)
					rowRaw := (*[1 << 30]uint16)(unsafe.Pointer(uintptr(ptr) + uintptr(t.offset) + uintptr(i*c*2)))[:c:c]
					for j := 0; j < c; j++ {
						row[j] = Float16ToFloat32(rowRaw[j])
					}
					dest[start+i] = row
				}
			} else {
				for i := s; i < e; i++ {
					row := make([]float32, c)
					rowRaw := (*[1 << 30]float32)(unsafe.Pointer(uintptr(ptr) + uintptr(t.offset) + uintptr(i*c*4)))[:c:c]
					copy(row, rowRaw)
					dest[start+i] = row
				}
			}
		}(wStart, wEnd)
	}
	wg.Wait()
}

func (t *MetalTensor) ExtractToFlat(dest []float32, start int) {
	ptr := C.Metal_GetBufferContents(t.buf)
	if ptr == nil {
		return
	}
	
	size := t.rows * t.cols
	if t.trans {
		data := t.ToHost()
		copy(dest[start:], data)
		return
	}

	if t.backend.useFP16 {
		raw := (*[1 << 30]uint16)(unsafe.Pointer(uintptr(ptr) + uintptr(t.offset)))[:size:size]
		// Explicit parallel conversion if large? 
		// For now simple loop
		for i := 0; i < size; i++ {
			dest[start+i] = Float16ToFloat32(raw[i])
		}
	} else {
		raw := (*[1 << 30]float32)(unsafe.Pointer(uintptr(ptr) + uintptr(t.offset)))[:size:size]
		copy(dest[start:], raw)
	}
}

func (t *MetalTensor) ApplyRoPE(batchSize, seqLen, numHeads, headDim int) {
	if t.backend.useFP16 {
		C.Metal_ApplyRoPE_F16(t.backend.ctx, t.buf, C.int(t.offset), C.int(batchSize), C.int(seqLen), C.int(numHeads), C.int(headDim))
	} else {
		panic("ApplyRoPE only supported for Metal FP16")
	}
}

func (t *MetalTensor) ExtractBytes() []byte {
	t.backend.Synchronize()
	
	size := t.rows * t.cols
	var sizeBytes int
	if t.dtype == Float16 {
		sizeBytes = size * 2
	} else {
		sizeBytes = size * 4
	}
	
	out := make([]byte, sizeBytes)
	C.Metal_ExtractBytes(t.buf, C.int(t.offset), unsafe.Pointer(&out[0]), C.int(sizeBytes))
	return out
}

func (t *MetalTensor) Cast(dtype DataType) Tensor {
	if t.dtype == dtype {
		// If backend default type != requested dtype?
		// NewTensor uses backend.useFP16. 
		// If we are casting, we likely want a new tensor that potentially differs from backend default.
		// We should manual construct.
		
		sizeBytes := t.sizeBytes
		buf := C.Metal_Alloc(t.backend.ctx, C.int(sizeBytes))
		C.Metal_CopyToDevice(buf, 0, unsafe.Pointer(uintptr(C.Metal_GetBufferContents(t.buf))+uintptr(t.offset)), C.int(sizeBytes)) // Hacky copy 
		// Actually better: use a kernel or Metal_CopyBuffer if we had it.
		// Or read/write.
		// Simplified: Just use existing copy logic if types match?
		// But CopyFrom expects tensor.
		
		// For now, let's implement the specific case we need: FP32 -> FP16.
		// Or FP16 -> FP16 (Copy)
		
		// If same type, just use NewTensor if backend matches type
		currentBackendType := Float32
		if t.backend.useFP16 { currentBackendType = Float16 }
		
		if dtype == currentBackendType {
			nt := t.backend.NewTensor(t.rows, t.cols, nil)
			nt.Copy(t)
			return nt
		}
	}
	
	if t.dtype == Float32 && dtype == Float16 {
		size := t.rows * t.cols
		outSizeBytes := size * 2
		outBuf := C.Metal_Alloc(t.backend.ctx, C.int(outSizeBytes))
		
		C.Metal_Cast_F32_to_F16(t.backend.ctx, t.buf, C.int(t.offset), outBuf, 0, C.int(size))
		
		tFinal := &MetalTensor{
			backend:    t.backend,
			rows:       t.rows,
			cols:       t.cols,
			buf:        outBuf,
			offset:     0,
			sizeBytes:  outSizeBytes,
			ownsBuffer: true,
			dtype:      Float16,
		}
		
		runtime.SetFinalizer(tFinal, func(mt *MetalTensor) {
			// We can't pool this easily if it mismatches backend buckets (FP16 vs FP32 sizes differ for same dims)
			// But returnToPool uses bytes size. So it works!
			C.Metal_FreeBuffer(mt.backend.ctx, mt.buf) // Or use pool?
			// Use free for safety for now on manual allocs
		})
		
		return tFinal
	}
	
	panic("Cast: Unsupported conversion")
}

func (b *MetalBackend) GetVRAMUsage() (int64, int64) {
	allocated := C.Metal_GetAllocatedSize(b.ctx)
	total := C.Metal_GetRecommendMaxWorkingSetSize(b.ctx)
	return int64(allocated), int64(total)
}

// FusedAttention performs scaled dot-product attention with all operations fused into a single kernel
// This is significantly faster than the unfused Attention() method due to reduced kernel dispatch overhead
// and better memory locality.
func (t *MetalTensor) FusedAttention(q, k, v Tensor, batchSize, seqLen int, scale float32) Tensor {
	if !t.backend.useFP16 {
		panic("FusedAttention requires FP16 backend")
	}
	
	qt := q.(*MetalTensor)
	kt := k.(*MetalTensor)
	vt := v.(*MetalTensor)
	
	r, c := qt.Dims()
	if r != batchSize*seqLen {
		panic("FusedAttention: dims mismatch")
	}
	
	result := t.backend.NewTensor(r, c, nil)
	rst := result.(*MetalTensor)
	
	C.Metal_FusedAttention_F16(t.backend.ctx,
		qt.buf, C.int(qt.offset),
		kt.buf, C.int(kt.offset),
		vt.buf, C.int(vt.offset),
		rst.buf, C.int(rst.offset),
		C.int(batchSize), C.int(seqLen), C.int(c), C.float(scale))
	
	return result
}
