package device

// Tensor represents a multi-dimensional array of data that can be resident
// on different devices (CPU, Metal GPU, CUDA GPU).
type Tensor interface {
	// Dims returns the dimensions of the tensor (rows, cols).
	Dims() (r, c int)
	
	// At returns the value at (i, j).
	// Note: This forces a sync to CPU if the tensor is on GPU, so use sparingly!
	At(i, j int) float64
	
	// Set sets the value at (i, j).
	// Note: Use sparingly for performance.
	Set(i, j int, v float64)
	
	// Copy transfers data from another tensor to this one.
	Copy(from Tensor)

	// Slice returns a view of the tensor.
	Slice(i, k, j, l int) Tensor

	// T returns the transpose of the tensor.
	// Depending on backend, this might be a view or a copy.
	T() Tensor

	// Mul performs matrix multiplication: receiver = a * b.
	Mul(a, b Tensor)
	
	// MulVec performs matrix-vector multiplication: result = matrix * vector
	// where the receiver is the result vector.
	// MulVec(m Tensor, v Tensor)

	// Add performs element-wise addition: receiver += other.
	Add(other Tensor)
	
	// AddScalar adds a scalar to each element of the tensor.
	AddScalar(val float64)
	
	// AddBias adds a bias vector to each row of the tensor.
	// len(bias) must equal tensor columns.
	AddBias(bias []float64)
	
	// Scale scales the tensor by a scalar.
	Scale(val float64)
	
	// Softmax applies Softmax function in-place row-wise.
	Softmax()
	
	// Gelu applies GELU activation in-place.
	Gelu()
	
	// Tanh applies Tanh activation in-place.
	Tanh()
	
	// LayerNorm applies Layer Normalization in-place.
	LayerNorm(gamma, beta Tensor, eps float64)
	
	// Gather gathers rows based on indices.
	Gather(indices []int) Tensor
	
	// ToHost copies the data back to a Go slice (row-major).
	ToHost() []float64

	// CopyFromFloat64 copies a []float64 slice to the tensor in bulk.
	// This is much more efficient than using Set() for each element.
	CopyFromFloat64(data []float64)

    // Data returns the underlying slice if on CPU, nil otherwise.
    // Dangerous, use with caution.
    Data() []float64
}

// Backend creates tensors and manages device memory.
type Backend interface {
	Name() string
	NewTensor(r, c int, data []float64) Tensor
	
	// GetTensor gets a tensor from the pool or creates a new one.
	GetTensor(r, c int) Tensor
	
	// PutTensor returns a tensor to the pool.
	PutTensor(t Tensor)
	
	Synchronize() // Block until all queued operations are complete
}
