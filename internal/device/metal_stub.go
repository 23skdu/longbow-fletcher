// +build !metal !darwin

package device

// MetalBackend is a stub for the Metal backend on unsupported platforms.
type MetalBackend struct{}

func NewMetalBackend() Backend {
	return &MetalBackend{}
}

func NewMetalBackendFP16() Backend {
	return &MetalBackend{}
}

func (b *MetalBackend) Name() string {
	return "Metal (stub)"
}

func (b *MetalBackend) NewTensor(r, c int, data []float32) Tensor {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) GetTensor(r, c int) Tensor {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) PutTensor(t Tensor) {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) Synchronize() {
	// No-op for stub
}

func (b *MetalBackend) DeviceCount() int {
	return 0
}

func (b *MetalBackend) SetDevice(index int) {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) GetVRAMUsage() (int64, int64) {
	return 0, 0
}

type MetalTensor struct{}

func (t *MetalTensor) Dims() (int, int)                                            { return 0, 0 }
func (t *MetalTensor) At(i, j int) float32                                         { return 0 }
func (t *MetalTensor) Set(i, j int, v float32)                                     {}
func (t *MetalTensor) Data() []float32                                             { return nil }
func (t *MetalTensor) ToHost() []float32                                           { return nil }
func (t *MetalTensor) CopyFromFloat32(data []float32)                              {}
func (t *MetalTensor) Copy(from Tensor)                                            {}
func (t *MetalTensor) Slice(i, k, j, l int) Tensor                                 { return nil }
func (t *MetalTensor) T() Tensor                                                   { return nil }
func (t *MetalTensor) Mul(a, b Tensor)                                             {}
func (t *MetalTensor) Add(other Tensor)                                            {}
func (t *MetalTensor) AddScalar(val float32)                                       {}
func (t *MetalTensor) Scale(val float32)                                           {}
func (t *MetalTensor) AddBias(bias Tensor)                                         {}
func (t *MetalTensor) Softmax()                                                    {}
func (t *MetalTensor) Gelu()                                                       {}
func (t *MetalTensor) Tanh()                                                       {}
func (t *MetalTensor) LayerNorm(gamma, beta Tensor, eps float32)                   {}
func (t *MetalTensor) Gather(indices []int) Tensor                                 { return nil }
func (t *MetalTensor) Linear(input, weight, bias Tensor) Tensor                    { return nil }
func (t *MetalTensor) LinearActivation(i, w, b Tensor, a ActivationType) Tensor    { return nil }
func (t *MetalTensor) Attention(q, k, v Tensor, b, s int, sc float32) Tensor       { return nil }
func (t *MetalTensor) ApplyRoPE(b, s, n, h int)                                    {}
func (t *MetalTensor) ExtractTo(dest [][]float32, start int)                       {}
func (t *MetalTensor) ExtractToFlat(dest []float32, start int)                     {}
func (t *MetalTensor) ExtractBytes() []byte                                        { return nil }
func (t *MetalTensor) Cast(dtype DataType) Tensor                                  { return nil }
