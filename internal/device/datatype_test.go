package device

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestDatatypeConsistency(t *testing.T) {
	tests := []struct {
		dtype    DataType
		wantSize int
		name     string
	}{
		{Float32, 4, "float32"},
		{Float16, 2, "float16"},
		{Float64, 8, "float64"},
		{Int8, 1, "int8"},
		{Int16, 2, "int16"},
		{Int32, 4, "int32"},
		{Int64, 8, "int64"},
		{Int, 8, "int"},
		{Uint8, 1, "uint8"},
		{Uint16, 2, "uint16"},
		{Uint32, 4, "uint32"},
		{Uint64, 8, "uint64"},
		{Uint, 8, "uint"},
		{Uintptr, 8, "uintptr"},
		{Complex64, 4, "complex64"},
		{Complex128, 8, "complex128"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.wantSize, DataTypeSize(tt.dtype))
		})
	}
}

func TestDimensionConsistency(t *testing.T) {
	expectedDims := []int{128, 384, 768, 1024, 1536, 2048, 3072}
	require.Equal(t, expectedDims, SupportedEmbeddingDimensions)

	for _, dim := range expectedDims {
		require.True(t, IsValidEmbeddingDimension(dim), "dimension %d should be valid", dim)
	}

	invalidDims := []int{0, -1, 1, 64, 256, 512, 1000, 1500, 4096, 8192}
	for _, dim := range invalidDims {
		require.False(t, IsValidEmbeddingDimension(dim), "dimension %d should be invalid", dim)
	}
}

func TestDataTypeString(t *testing.T) {
	tests := []struct {
		dtype DataType
		want  string
	}{
		{Float32, "float32"},
		{Float16, "float16"},
		{Float64, "float64"},
		{Int8, "int8"},
		{Int16, "int16"},
		{Int32, "int32"},
		{Int64, "int64"},
		{Int, "int"},
		{Uint8, "uint8"},
		{Uint16, "uint16"},
		{Uint32, "uint32"},
		{Uint64, "uint64"},
		{Uint, "uint"},
		{Uintptr, "uintptr"},
		{Complex64, "complex64"},
		{Complex128, "complex128"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			require.Equal(t, tt.want, tt.dtype.String())
		})
	}
}

func TestTensorAllDimensions(t *testing.T) {
	backend := NewCPUBackend()
	dimensions := []int{128, 384, 768, 1024, 1536, 2048, 3072}

	for _, dim := range dimensions {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			tensor := backend.NewTensor(1, dim, nil)

			r, c := tensor.Dims()
			require.Equal(t, 1, r)
			require.Equal(t, dim, c)

			for i := 0; i < dim; i++ {
				val := float32(i) * 0.1
				tensor.Set(0, i, val)
			}

			data := tensor.ToHost()
			require.Equal(t, dim, len(data))

			for i := 0; i < dim; i++ {
				expected := float32(i) * 0.1
				require.InDelta(t, expected, data[i], 0.001)
			}

			backend.PutTensor(tensor)
		})
	}
}

func TestTensorAllDatatypes(t *testing.T) {
	backend := NewCPUBackend()

	dtypes := []struct {
		dtype DataType
		size  int
	}{
		{Float32, 4},
		{Float16, 2},
		{Float64, 8},
		{Int8, 1},
		{Int16, 2},
		{Int32, 4},
		{Int64, 8},
		{Uint8, 1},
		{Uint16, 2},
		{Uint32, 4},
		{Uint64, 8},
	}

	for _, dt := range dtypes {
		t.Run(dt.dtype.String(), func(t *testing.T) {
			tensor := backend.NewTensorWithType(10, 10, dt.dtype, nil)

			r, c := tensor.Dims()
			require.Equal(t, 10, r)
			require.Equal(t, 10, c)

			require.Equal(t, dt.dtype, tensor.DataType())

			backend.PutTensor(tensor)
		})
	}
}

func TestTensorOperationsAllDimensions(t *testing.T) {
	backend := NewCPUBackend()
	dimensions := []int{128, 384, 768}

	for _, dim := range dimensions {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			a := backend.NewTensor(1, dim, nil)
			b := backend.NewTensor(1, dim, nil)

			for i := 0; i < dim; i++ {
				a.Set(0, i, float32(i+1))
				b.Set(0, i, float32(i+1))
			}

			a.Add(b)

			data := a.ToHost()
			for i := 0; i < dim; i++ {
				require.InDelta(t, float32(i+1)*2, data[i], 0.001)
			}

			a.Scale(0.5)
			data = a.ToHost()
			for i := 0; i < dim; i++ {
				require.InDelta(t, float32(i+1), data[i], 0.001)
			}

			backend.PutTensor(a)
			backend.PutTensor(b)
		})
	}
}

func TestMatrixMultiplyAllDimensions(t *testing.T) {
	backend := NewCPUBackend()
	testCases := []struct {
		m, n, k int
	}{
		{4, 4, 4},
		{8, 8, 8},
		{16, 16, 16},
		{32, 32, 32},
		{64, 64, 64},
		{128, 128, 128},
	}

	for _, tc := range testCases {
		t.Run("matmul_"+string(rune(tc.m)), func(t *testing.T) {
			a := backend.NewTensor(tc.m, tc.k, nil)
			b := backend.NewTensor(tc.k, tc.n, nil)
			c := backend.NewTensor(tc.m, tc.n, nil)

			for i := 0; i < tc.m; i++ {
				for j := 0; j < tc.k; j++ {
					a.Set(i, j, 1.0)
				}
			}
			for i := 0; i < tc.k; i++ {
				for j := 0; j < tc.n; j++ {
					b.Set(i, j, 1.0)
				}
			}

			c.Mul(a, b)

			data := c.ToHost()
			expected := float32(tc.k)
			for i := 0; i < tc.m; i++ {
				for j := 0; j < tc.n; j++ {
					idx := i*tc.n + j
					require.InDelta(t, expected, data[idx], 0.001, "at (%d,%d)", i, j)
				}
			}

			backend.PutTensor(a)
			backend.PutTensor(b)
			backend.PutTensor(c)
		})
	}
}

func TestLayerNormAllDimensions(t *testing.T) {
	backend := NewCPUBackend()
	dimensions := []int{64, 128, 256, 512, 768, 1024}

	for _, dim := range dimensions {
		t.Run("dim_"+string(rune(dim)), func(t *testing.T) {
			a := backend.NewTensor(1, dim, nil)
			gamma := backend.NewTensor(1, dim, nil)
			beta := backend.NewTensor(1, dim, nil)

			for i := 0; i < dim; i++ {
				a.Set(0, i, float32(i))
				gamma.Set(0, i, 1.0)
				beta.Set(0, i, 0.0)
			}

			a.LayerNorm(gamma, beta, 1e-6)

			data := a.ToHost()
			require.Len(t, data, dim)

			for _, v := range data {
				require.False(t, math.IsNaN(float64(v)), "LayerNorm output should not be NaN")
			}

			backend.PutTensor(a)
			backend.PutTensor(gamma)
			backend.PutTensor(beta)
		})
	}
}

func TestSoftmaxAllDimensions(t *testing.T) {
	backend := NewCPUBackend()
	sizes := []int{16, 32, 64, 128, 256, 512, 1024}

	for _, size := range sizes {
		t.Run("size_"+string(rune(size)), func(t *testing.T) {
			a := backend.NewTensor(1, size, nil)

			for i := 0; i < size; i++ {
				a.Set(0, i, float32(i))
			}

			a.Softmax()

			data := a.ToHost()

			var sum float32
			for _, v := range data {
				require.GreaterOrEqual(t, v, float32(0), "Softmax output should be non-negative")
				require.LessOrEqual(t, v, float32(1), "Softmax output should be <= 1")
				sum += v
			}

			require.InDelta(t, 1.0, float64(sum), 0.001, "Softmax sum should be ~1.0")

			backend.PutTensor(a)
		})
	}
}

func TestFuzzTensorCreation(t *testing.T) {
	backend := NewCPUBackend()

	for i := 0; i < 100; i++ {
		rows := rand.Intn(100) + 1
		cols := rand.Intn(100) + 1

		tensor := backend.NewTensor(rows, cols, nil)

		r, c := tensor.Dims()
		require.Equal(t, rows, r)
		require.Equal(t, cols, c)

		backend.PutTensor(tensor)
	}
}

func TestFuzzDimensionValidation(t *testing.T) {
	validDims := map[int]bool{
		128: true, 384: true, 768: true, 1024: true, 1536: true, 2048: true, 3072: true,
	}
	for i := 0; i < 1000; i++ {
		dim := rand.Intn(10000)
		_, isValid := validDims[dim]
		require.Equal(t, isValid, IsValidEmbeddingDimension(dim))
	}
}

func TestFuzzTensorOperations(t *testing.T) {
	backend := NewCPUBackend()

	for i := 0; i < 50; i++ {
		size := rand.Intn(200) + 10

		a := backend.NewTensor(1, size, nil)
		b := backend.NewTensor(1, size, nil)

		for j := 0; j < size; j++ {
			a.Set(0, j, rand.Float32()*10)
			b.Set(0, j, rand.Float32()*10)
		}

		a.Add(b)
		data := a.ToHost()
		require.Len(t, data, size)

		for _, v := range data {
			require.False(t, math.IsInf(float64(v), 0), "Add should not produce Inf")
		}

		a.Scale(2.0)
		data = a.ToHost()
		require.Len(t, data, size)

		a.Tanh()
		data = a.ToHost()
		require.Len(t, data, size)

		for _, v := range data {
			require.False(t, math.IsNaN(float64(v)), "Tanh should not produce NaN")
		}

		backend.PutTensor(a)
		backend.PutTensor(b)
	}
}

func TestFuzzNumericEdgeCases(t *testing.T) {
	backend := NewCPUBackend()

	edgeCases := []float32{
		0,
		math.MaxFloat32,
		math.SmallestNonzeroFloat32,
		math.MaxInt16,
		-math.MaxInt16,
	}

	for _, val := range edgeCases {
		t.Run("scale_"+string(rune(int(val))), func(t *testing.T) {
			a := backend.NewTensor(1, 10, nil)
			for i := 0; i < 10; i++ {
				a.Set(0, i, val)
			}

			a.Scale(2.0)
			data := a.ToHost()

			for _, v := range data {
				if val != 0 {
					require.Equal(t, val*2, v, "Scale should multiply correctly")
				}
			}

			backend.PutTensor(a)
		})
	}

	for _, val := range edgeCases {
		t.Run("add_"+string(rune(int(val))), func(t *testing.T) {
			a := backend.NewTensor(1, 10, nil)
			b := backend.NewTensor(1, 10, nil)
			for i := 0; i < 10; i++ {
				a.Set(0, i, val)
				b.Set(0, i, val)
			}

			a.Add(b)
			data := a.ToHost()

			for _, v := range data {
				require.Equal(t, val+val, v, "Add should add correctly")
			}

			backend.PutTensor(a)
			backend.PutTensor(b)
		})
	}
}

func TestPerformanceRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	backend := NewCPUBackend()

	start := time.Now()
	for i := 0; i < 1000; i++ {
		tensor := backend.NewTensor(1, 768, nil)
		data := tensor.ToHost()
		_ = data
		backend.PutTensor(tensor)
	}
	elapsed := time.Since(start)

	t.Logf("Created 1000 tensors in %v", elapsed)

	maxExpected := 5 * time.Second
	require.Less(t, elapsed, maxExpected, "Performance regression detected")
}
