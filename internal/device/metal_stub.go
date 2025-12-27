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
