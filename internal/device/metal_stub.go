// +build !metal !darwin

package device

func NewMetalBackend() Backend {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}

func NewMetalBackendFP16() Backend {
	panic("Metal backend is not supported on this platform. Build with -tags metal on macOS.")
}
