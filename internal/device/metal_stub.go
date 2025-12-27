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

func (b *MetalBackend) SetDevice(index int) {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) GetVRAMUsage() (int64, int64) {
	return 0, 0
}

func (b *MetalBackend) LoadWeights(weights []byte) error {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) Forward(input []float32) ([]float32, error) {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func (b *MetalBackend) Free() {
	// No-op for stub
}
