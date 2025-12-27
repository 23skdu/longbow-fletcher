//go:build !cuda

package device

func NewCudaBackend() Backend {
	panic("CUDA backend is not supported on this platform. Build with -tags cuda on Linux.")
}

func NewCudaBackendFP16() Backend {
	panic("CUDA backend is not supported on this platform. Build with -tags cuda on Linux.")
}
