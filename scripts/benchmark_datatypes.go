package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/23skdu/longbow-fletcher/internal/device"
)

var (
	dimensions = []int{128, 384, 768, 1024, 1536, 2048, 3072}
	datatypes  = []device.DataType{
		device.Float32,
		device.Float16,
		device.Float64,
		device.Int8,
		device.Int16,
		device.Int32,
		device.Int64,
		device.Uint8,
		device.Uint16,
		device.Uint32,
		device.Uint64,
		device.Complex64,
		device.Complex128,
	}
)

func BenchmarkDatatypeSizes(b *testing.B) {
	for _, dt := range datatypes {
		b.Run(dt.String(), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = device.DataTypeSize(dt)
			}
		})
	}
}

func BenchmarkTensorCreation(b *testing.B) {
	backend := device.NewCPUBackend()

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				tensor := backend.NewTensor(1, dim, nil)
				backend.PutTensor(tensor)
			}
		})
	}
}

func BenchmarkTensorOperations(b *testing.B) {
	backend := device.NewCPUBackend()

	ops := []struct {
		name string
		fn   func()
	}{
		{"Add", func() {
			a := backend.NewTensor(512, 512, nil)
			b := backend.NewTensor(512, 512, nil)
			a.Add(b)
			backend.PutTensor(a)
			backend.PutTensor(b)
		}},
		{"Scale", func() {
			a := backend.NewTensor(512, 512, nil)
			a.Scale(2.0)
			backend.PutTensor(a)
		}},
		{"Tanh", func() {
			a := backend.NewTensor(512, 512, nil)
			a.Tanh()
			backend.PutTensor(a)
		}},
		{"Softmax", func() {
			a := backend.NewTensor(1, 512, nil)
			a.Softmax()
			backend.PutTensor(a)
		}},
	}

	for _, op := range ops {
		b.Run(op.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				op.fn()
			}
		})
	}
}

func BenchmarkMatrixMultiply(b *testing.B) {
	backend := device.NewCPUBackend()

	sizes := []struct {
		m, n, k int
	}{
		{128, 128, 128},
		{256, 256, 256},
		{512, 512, 512},
		{768, 768, 768},
		{1024, 1024, 1024},
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d_%dx%d", size.m, size.k, size.k, size.n), func(b *testing.B) {
			a := backend.NewTensor(size.m, size.k, nil)
			c := backend.NewTensor(size.m, size.n, nil)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				c.Mul(a, a)
			}
			backend.PutTensor(a)
			backend.PutTensor(c)
		})
	}
}

func BenchmarkDimensionValidation(b *testing.B) {
	validDims := []int{128, 384, 768, 1024, 1536, 2048, 3072}
	invalidDims := []int{0, -1, 64, 256, 512, 1000, 4096}

	b.Run("ValidDimensions", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, dim := range validDims {
				_ = device.IsValidEmbeddingDimension(dim)
			}
		}
	})

	b.Run("InvalidDimensions", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, dim := range invalidDims {
				_ = device.IsValidEmbeddingDimension(dim)
			}
		}
	})
}

func BenchmarkMemoryAllocation(b *testing.B) {
	backend := device.NewCPUBackend()

	for _, dim := range dimensions {
		b.Run(fmt.Sprintf("dim_%d", dim), func(b *testing.B) {
			totalBytes := 0
			for i := 0; i < b.N; i++ {
				tensor := backend.NewTensor(1, dim, nil)
				_, size := tensor.Dims()
				totalBytes += size * 4 // float32
				backend.PutTensor(tensor)
			}
			b.SetBytes(int64(totalBytes))
		})
	}
}

func main() {
	flag.Parse()

	fmt.Printf("Running benchmarks on %d cores\n", runtime.NumCPU())
	fmt.Printf("Dimensions: %v\n", dimensions)
	fmt.Printf("Datatypes: %v\n", datatypes)

	// Run benchmarks
	fmt.Println("\n=== Datatype Size Benchmarks ===")
	testing.Benchmark(BenchmarkDatatypeSizes)

	fmt.Println("\n=== Tensor Creation Benchmarks ===")
	testing.Benchmark(BenchmarkTensorCreation)

	fmt.Println("\n=== Tensor Operations Benchmarks ===")
	testing.Benchmark(BenchmarkTensorOperations)

	fmt.Println("\n=== Matrix Multiply Benchmarks ===")
	testing.Benchmark(BenchmarkMatrixMultiply)

	fmt.Println("\n=== Dimension Validation Benchmarks ===")
	testing.Benchmark(BenchmarkDimensionValidation)

	fmt.Println("\n=== Memory Allocation Benchmarks ===")
	testing.Benchmark(BenchmarkMemoryAllocation)

	os.Exit(0)
}
