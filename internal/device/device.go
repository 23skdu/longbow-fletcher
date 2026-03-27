package device

// Tensor represents a multi-dimensional array of data that can be resident
// on different devices (CPU, Metal GPU, CUDA GPU).
type Tensor interface {
	// Dims returns the dimensions (rows, cols) of the tensor.
	Dims() (int, int)

	// At returns the value at (i, j).
	// This is often slow and should be used for debugging or infrequent access.
	At(i, j int) float32

	// Set sets the value at (i, j).
	Set(i, j int, v float32)

	// Data returns the underlying slice if available on CPU (nil if on GPU).
	Data() []float32

	// DataType returns the data type of the tensor.
	DataType() DataType

	// ToHost copies the data to a Go slice (float32).
	ToHost() []float32

	// CopyFromFloat32 copies data from a Go slice (float32) to the tensor.
	CopyFromFloat32(data []float32)

	// Operations

	// Copy copies content from another tensor.
	Copy(from Tensor)

	// Slice creates a view of the tensor.
	Slice(i, k, j, l int) Tensor

	// Paste copies data from a source tensor into this tensor at the specified position.
	// dstRow, dstCol: starting position in this tensor to paste into
	// src: source tensor to copy from
	// srcRow, srcCol: starting position in source tensor
	// rows, cols: number of rows and columns to copy
	Paste(dstRow, dstCol int, src Tensor, srcRow, srcCol, rows, cols int)

	// T returns the transpose view.
	T() Tensor

	// Mul performs matrix multiplication: result = this * other
	// In-place update of this tensor? No, usually Mul(a, b) -> writes to this.
	// Convention: t.Mul(a, b) means t = a * b
	Mul(a, b Tensor)

	// Add performs element-wise addition: t = t + other
	Add(other Tensor)

	// AddScalar performs: t = t + val
	AddScalar(val float32)

	// Scale performs: t = t * val
	Scale(val float32)

	// AddBias adds a bias vector (broadcasted) to each row/col.
	AddBias(bias Tensor)

	// Activation functions (In-Place)
	Softmax()
	Gelu()
	Tanh()

	// LayerNorm performs layer normalization (In-Place).
	LayerNorm(gamma, beta Tensor, eps float32)

	// AddLayerNorm performs fused Add + LayerNorm.
	// t = LayerNorm(t + residual).
	// t is updated in-place (containing the sum), then normalized.
	// Actually, usually residual is preserved?
	// In BERT: x = LayerNorm(x + residual)
	// If done in-place on x: x += residual; x = LayerNorm(x).
	// So this method modifies receiver 't'.
	AddLayerNorm(residual, gamma, beta Tensor, eps float32)

	// Gather collects rows based on indices. Returns new Tensor.
	Gather(indices []int) Tensor

	// Linear performs a fused MatMul + BiasAdd.
	// equivalent to: t.Mul(input, weight); t.AddBias(bias)
	// returns result tensor
	Linear(input, weight, bias Tensor) Tensor

	// LinearActivation performs Linear followed by Activation.
	LinearActivation(input, weight, bias Tensor, activation ActivationType) Tensor

	// Attention performs fused Scaled Dot Product Attention.
	// equivalent to: Softmax(Q * K^T * scale) * V
	// Assumes q, k, v are flattened (Batch*Seq, Hidden)
	// Returns flattend (Batch*Seq, Hidden)
	Attention(q, k, v Tensor, batchSize, seqLen, numHeads int, scale float32) Tensor

	// AttentionVarLen performs fused Scaled Dot Product Attention with variable sequence lengths.
	// equivalent to: Softmax(Q * K^T * scale) * V
	// Assumes q, k, v are flattened (Batch*Seq, Hidden)
	// 'lengths' specifies the actual sequence length for each batch item.
	// Returns flattend (Batch*Seq, Hidden)
	AttentionVarLen(q, k, v Tensor, lengths []int, numHeads int, scale float32) Tensor

	// RoPE applies Rotary Positional Embeddings to this tensor (In-Place).
	// Assumes tensor is (Batch*Seq, Hidden)
	ApplyRoPE(batchSize, seqLen, numHeads, headDim int)

	// ExtractTo parallelizes the transfer and row-splitting of the tensor into a pre-allocated slice of slices.
	ExtractTo(destination [][]float32, startRow int)

	// ExtractToFlat copies the tensor data into a pre-allocated flat slice.
	ExtractToFlat(destination []float32, startOffset int)

	// ExtractBytes returns the raw underlying byte representation of the tensor.
	// This is a copy of the data.
	ExtractBytes() []byte

	// Cast returns a new Tensor with the specified data type (cast performed on device).
	// returns new Tensor
	Cast(dtype DataType) Tensor

	// HasNaN checks for NaN values in the tensor.
	HasNaN() (bool, error)
}

type DataType int

const (
	// Float types
	Float32 DataType = iota
	Float16
	Float64

	// Signed integers
	Int8
	Int16
	Int32
	Int64
	Int

	// Unsigned integers
	Uint8
	Uint16
	Uint32
	Uint64
	Uint
	Uintptr

	// Complex types
	Complex64
	Complex128
)

// SupportedEmbeddingDimensions contains all valid embedding dimensions
var SupportedEmbeddingDimensions = []int{128, 384, 768, 1024, 1536, 2048, 3072}

func (d DataType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case Float64:
		return "float64"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Int:
		return "int"
	case Uint8:
		return "uint8"
	case Uint16:
		return "uint16"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Uint:
		return "uint"
	case Uintptr:
		return "uintptr"
	case Complex64:
		return "complex64"
	case Complex128:
		return "complex128"
	default:
		return "unknown"
	}
}

// IsValidEmbeddingDimension checks if the given dimension is supported
func IsValidEmbeddingDimension(dim int) bool {
	for _, d := range SupportedEmbeddingDimensions {
		if d == dim {
			return true
		}
	}
	return false
}

// DataTypeSize returns the byte size of each datatype
func DataTypeSize(dtype DataType) int {
	switch dtype {
	case Float32, Int32, Uint32, Complex64:
		return 4
	case Float16, Int16, Uint16:
		return 2
	case Float64, Int64, Uint64, Complex128:
		return 8
	case Int8, Uint8:
		return 1
	case Int, Uint:
		return 8 // 64-bit on 64-bit architecture
	case Uintptr:
		return 8
	default:
		return 4
	}
}

type ActivationType int

const (
	ActivationIdentity ActivationType = iota
	ActivationGELU
	ActivationTanh
	ActivationSoftmax // Usually not fused in Linear, but defined for completeness
	ActivationSwiGLU  // Fused Swish-Gated Linear Unit
)

// Backend creates tensors and manages device memory.
type Backend interface {
	Name() string
	NewTensor(r, c int, data []float32) Tensor
	NewTensorWithType(r, c int, dtype DataType, data []float32) Tensor

	// GetTensor gets a tensor from the pool or creates a new one.
	GetTensor(r, c int) Tensor

	// PutTensor returns a tensor to the pool.
	PutTensor(t Tensor)

	// Synchronize waits for all pending operations to complete.
	Synchronize()

	// DeviceCount returns the number of available devices.
	// 1 for CPU, >=1 for GPU.
	DeviceCount() int

	// SetDevice sets the current active device for this backend instance.
	// index must be < DeviceCount().
	SetDevice(index int)

	// GetVRAMUsage returns the currently allocated memory and total available memory (in bytes).
	// For CPU, this returns system memory stats.
	GetVRAMUsage() (allocated int64, total int64)
}
