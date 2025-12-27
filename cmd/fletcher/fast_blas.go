//go:build cgo
package main

// This file is only included when cgo is enabled.
// It registers the netlib BLAS implementation which uses system BLAS (Accelerate on macOS, OpenBLAS on Linux).

import (
	"github.com/rs/zerolog/log"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/netlib/blas/netlib"
)

func init() {
	// Register netlib BLAS
	blas64.Use(netlib.Implementation{})
	log.Debug().Msg("⚡ CGO/BLAS Acceleration Enabled (netlib)")
}

// Ensure gonum is also imported so we don't break non-cgo builds by accident if we mix things?
// No, the build tag handles it. But we need a fallback for non-cgo if we wanted to be robust.
// For now, let's just assume this file is for CGO speedup.
// Note: gonum/blas/gonum is the pure Go implementation used by default by gonum/mat if nothing else is registered.
// We override it here.

var _ = gonum.Implementation{} // Dummy usage to keep imports happy if needed? No need.
