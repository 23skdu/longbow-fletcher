package device

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQuantizedTensor(t *testing.T) {
	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	qt := QuantizeFloat32(data, 1, 128, QuantInt8, 32)
	require.NotNil(t, qt)

	r, c := qt.Dims()
	require.Equal(t, 1, r)
	require.Equal(t, 128, c)

	result := qt.ToFloat32()
	require.NotNil(t, result)
	require.Len(t, result, 128)
}

func TestQuantizedTensorInt4(t *testing.T) {
	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	qt := QuantizeFloat32(data, 1, 128, QuantInt4, 32)
	require.NotNil(t, qt)

	r, c := qt.Dims()
	require.Equal(t, 1, r)
	require.Equal(t, 128, c)

	result := qt.ToFloat32()
	require.NotNil(t, result)
	require.Len(t, result, 128)
}

func TestTurboQuant(t *testing.T) {
	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	config := TurboQuantConfig{
		BitWidth:  4,
		BlockSize: 32,
		UseQJL:    false,
	}

	qt := QuantizeTurbo(data, 128, config)
	require.NotNil(t, qt)

	r, c := qt.Dims()
	require.Equal(t, 1, r)
	require.Equal(t, 128, c)

	result := qt.ToFloat32()
	require.NotNil(t, result)
	require.Len(t, result, 128)

	ratio := qt.CompressionRatio()
	require.Greater(t, ratio, float64(1.0))
}

func TestPolarQuant(t *testing.T) {
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) * 0.1
	}

	polar := NewPolarQuantizer(32, 4)
	require.NotNil(t, polar)

	quantized, err := polar.Quantize(data)
	require.NoError(t, err)
	require.NotNil(t, quantized)

	decompressed := polar.Dequantize(quantized)
	require.NotNil(t, decompressed)
}

func TestTurboQuantMultipleDims(t *testing.T) {
	testDims := []int{64, 128, 256, 512, 1024}

	for _, dim := range testDims {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			data := make([]float32, dim)
			for i := range data {
				data[i] = float32(i) * 0.1
			}

			config := TurboQuantConfig{
				BitWidth:  4,
				BlockSize: 64,
				UseQJL:    false,
			}

			qt := QuantizeTurbo(data, dim, config)
			require.NotNil(t, qt)

			result := qt.ToFloat32()
			require.NotNil(t, result)
		})
	}
}

func TestTurboQuantMultipleBitWidths(t *testing.T) {
	bitWidths := []int{2, 4, 6, 8}

	for _, bits := range bitWidths {
		t.Run("bits_"+string(rune(bits)), func(t *testing.T) {
			data := make([]float32, 128)
			for i := range data {
				data[i] = float32(i) * 0.1
			}

			config := TurboQuantConfig{
				BitWidth:  bits,
				BlockSize: 32,
				UseQJL:    false,
			}

			qt := QuantizeTurbo(data, 128, config)
			require.NotNil(t, qt)

			ratio := qt.CompressionRatio()
			t.Logf("BitWidth %d: compression ratio %.2fx", bits, ratio)
		})
	}
}

func TestQuantizationTypes(t *testing.T) {
	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i)
	}

	qt := QuantizeFloat32(data, 1, 128, QuantInt8, 32)
	require.NotNil(t, qt)
}
