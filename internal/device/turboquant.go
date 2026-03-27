package device

import (
	"math"
	"math/rand"
)

type TurboQuantConfig struct {
	BitWidth      int
	BlockSize     int
	UseQJL        bool
	QJLDimensions int
}

const (
	TurboQuantDefaultBlockSize = 64
	TurboQuantDefaultBits      = 4
)

type TurboQuantizer struct {
	config TurboQuantConfig
	scales []float32
	zeros  []float32
	random *rand.Rand

	qjlMatrix []float32
}

func NewTurboQuantizer(config TurboQuantConfig) *TurboQuantizer {
	if config.BlockSize <= 0 {
		config.BlockSize = TurboQuantDefaultBlockSize
	}
	if config.BitWidth <= 0 {
		config.BitWidth = TurboQuantDefaultBits
	}
	if config.QJLDimensions <= 0 {
		config.QJLDimensions = 64
	}

	return &TurboQuantizer{
		config: config,
		scales: nil,
		zeros:  nil,
		random: rand.New(rand.NewSource(42)),
	}
}

func (t *TurboQuantizer) Quantize(data []float32, dims int) ([]byte, error) {
	numBlocks := (dims + t.config.BlockSize - 1) / t.config.BlockSize
	t.scales = make([]float32, numBlocks)
	t.zeros = make([]float32, numBlocks)

	quantizedData := make([]byte, numBlocks*t.config.BlockSize*t.config.BitWidth/8*2)

	blockIdx := 0
	bits := t.config.BitWidth
	for start := 0; start < dims && blockIdx < numBlocks; start += t.config.BlockSize {
		end := start + t.config.BlockSize
		if end > dims {
			end = dims
		}

		block := data[start:end]
		radius, angles := t.cartesianToPolar(block)

		quantizedRadius, scale, zero := t.quantizeValue(radius, bits)
		t.scales[blockIdx] = scale
		t.zeros[blockIdx] = zero

		offset := blockIdx * (t.config.BlockSize / 8 * t.config.BitWidth)
		quantizedData[offset] = byte(quantizedRadius)

		for i, angle := range angles {
			qAngle, _, _ := t.quantizeValue(angle, bits)
			angleOffset := offset + 1 + i*(t.config.BitWidth/8)
			if angleOffset < len(quantizedData) {
				quantizedData[angleOffset] = byte(qAngle)
			}
		}

		blockIdx++
	}

	if t.config.UseQJL {
		t.computeQJLMatrix(dims)
	}

	return quantizedData, nil
}

func (t *TurboQuantizer) Dequantize(quantizedData []float32, dims int) []float32 {
	result := make([]float32, dims)
	numBlocks := (dims + t.config.BlockSize - 1) / t.config.BlockSize

	blockIdx := 0
	for start := 0; start < dims && blockIdx < numBlocks; start += t.config.BlockSize {
		end := start + t.config.BlockSize
		if end > dims {
			end = dims
		}
		blockSize := end - start

		scale := t.scales[blockIdx]
		zero := t.zeros[blockIdx]

		offset := blockIdx * (t.config.BlockSize / 8 * t.config.BitWidth)
		radius := t.dequantizeValue(float32(quantizedData[offset]), scale, zero)

		polarCoords := make([]float32, blockSize)
		polarCoords[0] = radius

		for i := 1; i < blockSize && offset+1+i < len(quantizedData); i++ {
			angleOffset := offset + 1 + i
			if angleOffset < len(quantizedData) {
				polarCoords[i] = t.dequantizeValue(float32(quantizedData[angleOffset]), scale, zero)
			}
		}

		cartesian := t.polarToCartesian(polarCoords)
		copy(result[start:end], cartesian)

		blockIdx++
	}

	if t.config.UseQJL && t.qjlMatrix != nil {
		result = t.applyQJLCorrection(result, dims)
	}

	return result
}

func (t *TurboQuantizer) cartesianToPolar(data []float32) (float32, []float32) {
	if len(data) < 2 {
		if len(data) == 1 {
			val := float32(math.Abs(float64(data[0])))
			return val, []float32{0}
		}
		return 0, []float32{}
	}

	radius := float32(0)
	for _, v := range data {
		radius += v * v
	}
	radius = float32(math.Sqrt(float64(radius)))

	angles := make([]float32, len(data))
	if radius > 1e-6 {
		for i, v := range data {
			angles[i] = float32(math.Atan2(float64(v), float64(data[0])))
		}
	}

	return radius, angles
}

func (t *TurboQuantizer) polarToCartesian(polar []float32) []float32 {
	result := make([]float32, len(polar))
	if len(polar) == 0 {
		return result
	}

	radius := polar[0]
	for i := 1; i < len(polar); i++ {
		result[i] = radius * float32(math.Sin(float64(polar[i])))
	}
	if len(polar) > 0 {
		result[0] = radius * float32(math.Cos(float64(polar[1])))
	}

	return result
}

func (t *TurboQuantizer) quantizeValue(val float32, bits int) (int, float32, float32) {
	shifted := 1 << uint(bits)
	maxVal := float32(shifted - 1)
	scale := maxVal / (math.MaxFloat32 / 2)
	zero := 0.0

	scaled := (float64(val) + zero) / float64(scale)
	quantized := int(scaled + 0.5)
	if quantized < 0 {
		quantized = 0
	}
	maxQuantized := shifted - 1
	if quantized > maxQuantized {
		quantized = maxQuantized
	}

	return quantized, scale, float32(zero)
}

func (t *TurboQuantizer) dequantizeValue(val float32, scale, zero float32) float32 {
	return val*scale - zero
}

func (t *TurboQuantizer) computeQJLMatrix(dims int) {
	t.qjlMatrix = make([]float32, dims*t.config.QJLDimensions)
	for i := 0; i < dims*t.config.QJLDimensions; i++ {
		t.qjlMatrix[i] = float32(t.random.NormFloat64()) / float32(math.Sqrt(float64(t.config.QJLDimensions)))
	}
}

func (t *TurboQuantizer) applyQJLCorrection(data []float32, dims int) []float32 {
	if t.qjlMatrix == nil {
		return data
	}

	compressed := make([]float32, t.config.QJLDimensions)
	for j := 0; j < t.config.QJLDimensions; j++ {
		var sum float32
		for i := 0; i < dims; i++ {
			sum += data[i] * t.qjlMatrix[i*t.config.QJLDimensions+j]
		}
		if math.Signbit(float64(sum)) {
			compressed[j] = -1
		} else {
			compressed[j] = 1
		}
	}

	reconstructed := make([]float32, dims)
	for i := 0; i < dims; i++ {
		var sum float32
		for j := 0; j < t.config.QJLDimensions; j++ {
			sum += compressed[j] * t.qjlMatrix[i*t.config.QJLDimensions+j]
		}
		reconstructed[i] = data[i] - sum*0.1
	}

	return reconstructed
}

type TurboQuantizedTensor struct {
	data      []byte
	scales    []float32
	zeros     []float32
	dims      int
	blockSize int
	bitWidth  int
	useQJL    bool
}

func (t *TurboQuantizedTensor) Dims() (int, int) {
	return 1, t.dims
}

func (t *TurboQuantizedTensor) ToFloat32() []float32 {
	quantizer := TurboQuantConfig{
		BitWidth:  t.bitWidth,
		BlockSize: t.blockSize,
		UseQJL:    t.useQJL,
	}
	q := NewTurboQuantizer(quantizer)
	q.scales = t.scales
	q.zeros = t.zeros

	floatData := make([]float32, len(t.data))
	for i, b := range t.data {
		floatData[i] = float32(b)
	}

	return q.Dequantize(floatData, t.dims)
}

func QuantizeTurbo(data []float32, dims int, config TurboQuantConfig) *TurboQuantizedTensor {
	quantizer := NewTurboQuantizer(config)

	quantizedData, err := quantizer.Quantize(data, dims)
	if err != nil {
		panic(err)
	}

	return &TurboQuantizedTensor{
		data:      quantizedData,
		scales:    quantizer.scales,
		zeros:     quantizer.zeros,
		dims:      dims,
		blockSize: config.BlockSize,
		bitWidth:  config.BitWidth,
		useQJL:    config.UseQJL,
	}
}

func (t *TurboQuantizedTensor) CompressionRatio() float64 {
	originalSize := float64(t.dims * 4)
	quantizedSize := float64(len(t.data))
	return originalSize / quantizedSize
}

type PolarQuantizer struct {
	scales    []float32
	zeros     []float32
	blockSize int
	bitWidth  int
}

func NewPolarQuantizer(blockSize, bitWidth int) *PolarQuantizer {
	return &PolarQuantizer{
		blockSize: blockSize,
		bitWidth:  bitWidth,
	}
}

func (p *PolarQuantizer) Quantize(data []float32) ([]byte, error) {
	numBlocks := (len(data) + p.blockSize - 1) / p.blockSize
	p.scales = make([]float32, numBlocks)
	p.zeros = make([]float32, numBlocks)

	quantizedData := make([]byte, 0, numBlocks*(p.blockSize/8*p.bitWidth+1))

	bitWidthVal := p.bitWidth
	maxQuantizedVal := (1 << uint(bitWidthVal)) - 1
	maxFloatVal := float32(maxQuantizedVal)

	for b := 0; b < numBlocks; b++ {
		start := b * p.blockSize
		end := start + p.blockSize
		if end > len(data) {
			end = len(data)
		}

		block := data[start:end]
		radius, angles := p.cartesianToPolar(block)

		scale := radius / maxFloatVal
		if scale < 1e-6 {
			scale = 1e-6
		}
		p.scales[b] = scale
		p.zeros[b] = 0

		quantizedRadius := uint8(radius / scale)
		quantizedData = append(quantizedData, quantizedRadius)

		for _, angle := range angles {
			angleVal := (angle + math.Pi) / (2 * math.Pi) * maxFloatVal
			quantizedAngle := uint8(angleVal)
			quantizedData = append(quantizedData, quantizedAngle)
		}
	}

	return quantizedData, nil
}

func (p *PolarQuantizer) Dequantize(quantizedData []byte) []float32 {
	result := make([]float32, 0, len(p.scales)*p.blockSize)

	bitWidthVal := p.bitWidth
	maxQuantizedVal := (1 << uint(bitWidthVal)) - 1
	maxFloatVal := float32(maxQuantizedVal)

	idx := 0
	for b := 0; b < len(p.scales); b++ {
		radius := float32(quantizedData[idx]) * p.scales[b]
		idx++

		for i := 0; i < p.blockSize && idx < len(quantizedData); i++ {
			angle := (float32(quantizedData[idx]) / maxFloatVal) * 2 * math.Pi
			idx++

			val := radius * float32(math.Cos(float64(angle)))
			if i > 0 && idx-1 < len(quantizedData) {
				val = radius * float32(math.Sin(float64(angle)))
			}
			result = append(result, val)
		}
	}

	return result
}

func (p *PolarQuantizer) cartesianToPolar(data []float32) (float32, []float32) {
	radius := float32(0)
	for _, v := range data {
		radius += v * v
	}
	radius = float32(math.Sqrt(float64(radius)))

	angles := make([]float32, len(data))
	if radius > 1e-6 && len(data) > 0 {
		for i := range data {
			angles[i] = float32(math.Atan2(float64(data[i]), float64(data[0])))
		}
	}

	return radius, angles
}

type QJLQuantizer struct {
	dimensions int
	random     *rand.Rand
	matrix     []float32
}

func NewQJLQuantizer(dimensions int) *QJLQuantizer {
	q := &QJLQuantizer{
		dimensions: dimensions,
		random:     rand.New(rand.NewSource(42)),
	}
	q.generateMatrix()
	return q
}

func (q *QJLQuantizer) generateMatrix() {
	q.matrix = make([]float32, q.dimensions*q.dimensions)
	for i := 0; i < q.dimensions*q.dimensions; i++ {
		q.matrix[i] = float32(q.random.NormFloat64()) / float32(math.Sqrt(float64(q.dimensions)))
	}
}

func (q *QJLQuantizer) Compress(data []float32) []int8 {
	result := make([]int8, q.dimensions)

	for j := 0; j < q.dimensions; j++ {
		var dotProduct float32
		for i := 0; i < len(data); i++ {
			dotProduct += data[i] * q.matrix[i*q.dimensions+j]
		}
		if dotProduct >= 0 {
			result[j] = 1
		} else {
			result[j] = -1
		}
	}

	return result
}

func (q *QJLQuantizer) Decompress(compressed []int8) []float32 {
	reconstructed := make([]float32, len(q.matrix)/q.dimensions)

	for i := 0; i < len(reconstructed); i++ {
		var sum float32
		for j := 0; j < q.dimensions && i*q.dimensions+j < len(q.matrix); j++ {
			sum += float32(compressed[j]) * q.matrix[i*q.dimensions+j]
		}
		reconstructed[i] = sum * float32(math.Sqrt(float64(q.dimensions)))
	}

	return reconstructed
}
