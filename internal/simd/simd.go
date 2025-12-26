package simd

// ExpFast is a fast approximation of exp(x)
// Uses the identity exp(x) = 2^(x/ln2) and a polynomial approximation
func ExpFast(x float64) float64 {
	// Clamp to avoid overflow
	if x > 88 {
		return 1e38
	}
	if x < -88 {
		return 0
	}
	
	// exp(x) = 2^(x * log2(e))
	// log2(e) ≈ 1.4426950408889634
	const log2e = 1.4426950408889634
	
	t := x * log2e
	k := int(t)
	if t < 0 {
		k--
	}
	
	// Fractional part in [0, 1)
	f := t - float64(k)
	
	// Polynomial approximation for 2^f where f in [0, 1)
	// 2^f ≈ 1 + f*ln(2) + f^2*ln(2)^2/2 + ...
	// Simplified: 2^f ≈ 1 + 0.6931*f + 0.2401*f^2 + 0.0554*f^3
	p := 1.0 + f*(0.6931471805599453+f*(0.24022650695910072+f*0.05550410866482157))
	
	// Multiply by 2^k using bit manipulation
	if k >= 0 && k < 1024 {
		return p * float64(uint64(1)<<k)
	}
	if k < 0 && k > -1024 {
		return p / float64(uint64(1)<<(-k))
	}
	return p
}

// TanhFast is a fast approximation of tanh(x)
func TanhFast(x float64) float64 {
	// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
	// For |x| > 4, tanh approaches ±1
	if x > 4 {
		return 1
	}
	if x < -4 {
		return -1
	}
	
	// Rational approximation for small x
	x2 := x * x
	// Padé approximation: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)
	return x * (27.0 + x2) / (27.0 + 9.0*x2)
}

// GeluFast applies fast GELU approximation in-place
func GeluFast(data []float64) {
	const (
		sqrt2overPi = 0.7978845608
		coeff       = 0.044715
	)
	for i, x := range data {
		// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		data[i] = 0.5 * x * (1 + TanhFast(sqrt2overPi*(x+coeff*x*x*x)))
	}
}

// SoftmaxFast applies fast softmax in-place to a row
func SoftmaxFast(row []float64) {
	// Find max
	max := row[0]
	for _, v := range row {
		if v > max {
			max = v
		}
	}
	
	// Exp and sum using fast exp
	var sum float64
	for i, v := range row {
		row[i] = ExpFast(v - max)
		sum += row[i]
	}
	
	// Normalize
	invSum := 1.0 / sum
	for i := range row {
		row[i] *= invSum
	}
}

// VecAdd performs dst += src for float64 vectors
func VecAdd(dst, src []float64) {
	// Unrolled loop for better pipelining
	i := 0
	for ; i <= len(dst)-4; i += 4 {
		dst[i] += src[i]
		dst[i+1] += src[i+1]
		dst[i+2] += src[i+2]
		dst[i+3] += src[i+3]
	}
	// Handle remainder
	for ; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

// VecAddScaled performs dst += src * scale for float64 vectors
func VecAddScaled(dst, src []float64, scale float64) {
	i := 0
	for ; i <= len(dst)-4; i += 4 {
		dst[i] += src[i] * scale
		dst[i+1] += src[i+1] * scale
		dst[i+2] += src[i+2] * scale
		dst[i+3] += src[i+3] * scale
	}
	for ; i < len(dst); i++ {
		dst[i] += src[i] * scale
	}
}

// DotProduct computes the dot product of two float64 vectors
func DotProduct(a, b []float64) float64 {
	var sum float64
	i := 0
	for ; i <= len(a)-4; i += 4 {
		sum += a[i] * b[i]
		sum += a[i+1] * b[i+1]
		sum += a[i+2] * b[i+2]
		sum += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// MatVecMul performs dst = mat * vec where mat is rows x cols row-major
func MatVecMul(dst []float64, mat []float64, vec []float64, rows, cols int) {
	// For small matrices, straightforward implementation is fine
	// But we can unroll the inner loop
	for i := 0; i < rows; i++ {
		rowStart := i * cols
		row := mat[rowStart : rowStart+cols]
		dst[i] = DotProduct(row, vec)
	}
}
