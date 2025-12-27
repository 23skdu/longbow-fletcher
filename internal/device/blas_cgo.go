//go:build cgo

package device

// This file registers the netlib BLAS implementation which uses system BLAS
// (Accelerate on macOS, OpenBLAS on Linux) when CGO is available.

import (
	"github.com/rs/zerolog/log"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/netlib/blas/netlib"
)

func init() {
	// Register netlib BLAS for float32 operations (sgemm, etc.)
	blas32.Use(netlib.Implementation{})
	log.Debug().Msg("⚡ CGO/BLAS Acceleration Enabled (netlib)")
}
