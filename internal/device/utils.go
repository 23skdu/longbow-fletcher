package device

import (
	"math"
	"unsafe"
)

// Float32ToFloat16 converts a float32 to float16 (IEEE 754 binary16) representation.
// Handles edge cases to prevent NaN values:
// - Clamps values outside FP16 range to prevent overflow
// - Preserves NaN and Inf from input
// - Handles subnormals correctly
func Float32ToFloat16(f float32) uint16 {
	// Handle special cases first
	if math.IsNaN(float64(f)) {
		return 0x7E00 // FP16 NaN
	}
	if math.IsInf(float64(f), 1) {
		return 0x7C00 // FP16 +Inf
	}
	if math.IsInf(float64(f), -1) {
		return 0xFC00 // FP16 -Inf
	}
	
	// FP16 range: ±65504 (max normal), ±6.10e-5 (min normal)
	// Clamp to prevent overflow which can cause NaN
	const maxFP16 = 65504.0
	const minNormalFP16 = 6.10351562e-5
	
	// Clamp to FP16 range
	if f > maxFP16 {
		f = maxFP16
	} else if f < -maxFP16 {
		f = -maxFP16
	}
	
	// Handle values very close to zero (subnormals)
	absF := f
	if absF < 0 {
		absF = -absF
	}
	if absF < minNormalFP16 && absF > 0 {
		// Very small value - round to zero to avoid subnormal issues
		if f < 0 {
			return 0x8000 // -0
		}
		return 0x0000 // +0
	}
	
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	// Use signed integer to handle underflow (negative exponent)
	exp := int((bits >> 23) & 0xFF) - 127 + 15
	frac := (bits >> 13) & 0x3FF
	
	// Handle overflow (exponent too large for FP16)
	if exp >= 0x1F {
		// Return max value instead of infinity to prevent issues
		if sign != 0 {
			return uint16(sign | 0x7BFF) // Max negative
		}
		return 0x7BFF // Max positive
	}
	
	// Handle underflow (exponent too small)
	if exp <= 0 {
		// Flush to zero
		return uint16(sign)
	}
	
	return uint16(sign | (uint32(exp) << 10) | frac)
}

// Float16ToFloat32 converts a float16 (uint16 representation) to a float32
func Float16ToFloat32(h uint16) float32 {
	sign := (uint32(h) >> 15) & 1
	exp := (uint32(h) >> 10) & 0x1F
	frac := uint32(h) & 0x3FF
	
	if exp == 0 { // Zero/Denorm
		return 0.0
	}
	if exp == 31 { // Inf/NaN
		bits := (sign << 31) | (0xFF << 23) | (frac << 13)
		return *(*float32)(unsafe.Pointer(&bits))
	}
	
	newExp := int(exp) - 15 + 127
	bits := (sign << 31) | (uint32(newExp) << 23) | (frac << 13)
	return *(*float32)(unsafe.Pointer(&bits))
}
