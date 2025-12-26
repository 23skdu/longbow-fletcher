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
	
	// ToHost copies the data to a Go slice (float32).
	ToHost() []float32
	
	// CopyFromFloat32 copies data from a Go slice (float32) to the tensor.
	CopyFromFloat32(data []float32)
	
	// Operations
	
	// Copy copies content from another tensor.
	Copy(from Tensor)
	
	// Slice creates a view of the tensor.
	Slice(i, k, j, l int) Tensor
	
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
	Attention(q, k, v Tensor, batchSize, seqLen int, scale float32) Tensor

	// RoPE applies Rotary Positional Embeddings to this tensor (In-Place).
	// Assumes tensor is (Batch*Seq, Hidden)
	ApplyRoPE(batchSize, seqLen, numHeads, headDim int)
	
	// ExtractTo parallelizes the transfer and row-splitting of the tensor into a pre-allocated slice of slices.
	ExtractTo(destination [][]float32, startRow int)
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
	
	// GetTensor gets a tensor from the pool or creates a new one.
	GetTensor(r, c int) Tensor
	
	// PutTensor returns a tensor to the pool.
	PutTensor(t Tensor)
	
	// Synchronize() // Block until all queued operations are complete
	Synchronize()
}
