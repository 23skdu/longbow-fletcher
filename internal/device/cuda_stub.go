//go:build !cuda

package device

type CudaBackend struct{}

func NewCudaBackend() Backend {
	panic("CUDA backend is not supported on this platform. Build with -tags cuda on Linux.")
}

func (b *CudaBackend) SetDevice(index int) {
	panic("Not implemented on this platform")
}

func (b *CudaBackend) GetVRAMUsage() (int64, int64) {
	return 0, 0
}

func NewCudaBackendFP16() Backend {
	panic("CUDA backend is not supported on this platform. Build with -tags cuda on Linux.")
}
