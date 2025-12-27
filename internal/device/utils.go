package device

import (
	"unsafe"
)

// Float32ToFloat16 converts a float32 to a float16 (uint16 representation)
func Float32ToFloat16(f float32) uint16 {
	bits := *(*uint32)(unsafe.Pointer(&f))
	sign := (bits >> 31) & 1
	exp := (bits >> 23) & 0xFF
	frac := bits & 0x7FFFFF
	
	if exp == 255 { // Inf/NaN
		return uint16((sign << 15) | (0x1F << 10) | (frac >> 13))
	}
	if exp == 0 { // Zero/Denorm
		return uint16(sign << 15)
	}
	
	newExp := int(exp) - 127 + 15
	if newExp >= 31 { // Overflow -> Inf
		return uint16((sign << 15) | (0x1F << 10))
	}
	if newExp <= 0 { // Underflow -> Zero
		return uint16(sign << 15)
	}
	
	return uint16((sign << 15) | (uint32(newExp) << 10) | (frac >> 13))
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
