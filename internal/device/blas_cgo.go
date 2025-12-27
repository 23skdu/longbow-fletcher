//go:build cgo

package device

// This file registers the netlib BLAS implementation which uses system BLAS
// (Accelerate on macOS, OpenBLAS on Linux) when CGO is available.

import (
	"log"

	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/netlib/blas/netlib"
)

func init() {
	// Register netlib BLAS for float32 operations (sgemm, etc.)
	blas32.Use(netlib.Implementation{})
	log.Println("⚡ CGO/BLAS Acceleration Enabled (netlib)")
}
