package device

import (
	"testing"
	"time"
)

// BenchmarkMetalMatMul benchmarks matrix multiplication performance
// This establishes baseline for Step 1: Metal Kernel Optimization
func BenchmarkMetalMatMul(b *testing.B) {
	backend := NewMetalBackend()
	
	sizes := []struct {
		name string
		m, k, n int
	}{
		{"Small_128x128", 128, 128, 128},
		{"Medium_512x512", 512, 512, 512},
		{"Large_1024x1024", 1024, 1024, 1024},
		{"Tall_2048x128", 2048, 128, 128},    // Typical embedding projection
		{"Wide_128x2048", 128, 128, 2048},    // Typical MLP expansion
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create test matrices
			A := backend.NewTensor(size.m, size.k, nil)
			B := backend.NewTensor(size.k, size.n, nil)
			C := backend.NewTensor(size.m, size.n, nil)
			
			// Initialize with random data
			aData := make([]float32, size.m*size.k)
			bData := make([]float32, size.k*size.n)
			for i := range aData {
				aData[i] = float32(i % 100) / 100.0
			}
			for i := range bData {
				bData[i] = float32(i % 100) / 100.0
			}
			A.CopyFromFloat32(aData)
			B.CopyFromFloat32(bData)
			
			// Warm-up
			C.Mul(A, B)
			backend.Synchronize()
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				C.Mul(A, B)
			}
			backend.Synchronize()
			
			// Calculate GFLOPS
			ops := 2 * int64(size.m) * int64(size.k) * int64(size.n) // 2 ops per multiply-add
			elapsed := b.Elapsed()
			gflops := float64(ops*int64(b.N)) / float64(elapsed.Nanoseconds())
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkMetalLayerNorm benchmarks layer normalization performance
func BenchmarkMetalLayerNorm(b *testing.B) {
	backend := NewMetalBackend()
	
	sizes := []struct {
		name string
		batch, dim int
	}{
		{"Small_32x128", 32, 128},
		{"Medium_256x384", 256, 384},
		{"Large_512x768", 512, 768},
		{"XLarge_1024x768", 1024, 768},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			data := make([]float32, size.batch*size.dim)
			for i := range data {
				data[i] = float32(i%100) / 10.0
			}
			
			tensor := backend.NewTensor(size.batch, size.dim, data)
			gamma := backend.NewTensor(1, size.dim, nil)
			beta := backend.NewTensor(1, size.dim, nil)
			
			// Initialize gamma to 1, beta to 0
			gammaData := make([]float32, size.dim)
			betaData := make([]float32, size.dim)
			for i := range gammaData {
				gammaData[i] = 1.0
			}
			gamma.CopyFromFloat32(gammaData)
			beta.CopyFromFloat32(betaData)
			
			// Warm-up
			tensor.LayerNorm(gamma, beta, 1e-5)
			backend.Synchronize()
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				tensor.LayerNorm(gamma, beta, 1e-5)
			}
			backend.Synchronize()
			
			// Calculate throughput (elements/sec)
			elements := int64(size.batch) * int64(size.dim)
			elapsed := b.Elapsed()
			throughput := float64(elements*int64(b.N)) / elapsed.Seconds() / 1e9
			b.ReportMetric(throughput, "Gelems/s")
		})
	}
}

// BenchmarkMetalAttention benchmarks attention mechanism performance
func BenchmarkMetalAttention(b *testing.B) {
	backend := NewMetalBackendFP16() // Attention requires FP16
	
	sizes := []struct {
		name string
		batch, seq, hidden int
	}{
		{"Small_1x64x128", 1, 64, 128},
		{"Medium_8x128x384", 8, 128, 384},
		{"Large_16x256x768", 16, 256, 768},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			totalDim := size.batch * size.seq
			
			q := backend.NewTensor(totalDim, size.hidden, nil)
			k := backend.NewTensor(totalDim, size.hidden, nil)
			v := backend.NewTensor(totalDim, size.hidden, nil)
			
			// Initialize with random data
			qData := make([]float32, totalDim*size.hidden)
			kData := make([]float32, totalDim*size.hidden)
			vData := make([]float32, totalDim*size.hidden)
			for i := range qData {
				qData[i] = float32(i%100) / 100.0
				kData[i] = float32(i%100) / 100.0
				vData[i] = float32(i%100) / 100.0
			}
			q.CopyFromFloat32(qData)
			k.CopyFromFloat32(kData)
			v.CopyFromFloat32(vData)
			
			scale := float32(1.0 / float32(size.hidden))
			
			// Warm-up
			result := q.Attention(q, k, v, size.batch, size.seq, scale)
			backend.Synchronize()
			backend.PutTensor(result)
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				result = q.Attention(q, k, v, size.batch, size.seq, scale)
				backend.PutTensor(result)
			}
			backend.Synchronize()
			
			// Calculate GFLOPS (attention is O(seq^2 * hidden))
			ops := int64(size.batch) * int64(size.seq) * int64(size.seq) * int64(size.hidden) * 2
			elapsed := b.Elapsed()
			gflops := float64(ops*int64(b.N)) / float64(elapsed.Nanoseconds())
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}

// BenchmarkMetalSoftmax benchmarks softmax performance
func BenchmarkMetalSoftmax(b *testing.B) {
	backend := NewMetalBackend()
	
	sizes := []struct {
		name string
		rows, cols int
	}{
		{"Small_32x128", 32, 128},
		{"Medium_256x512", 256, 512},
		{"Large_1024x1024", 1024, 1024},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			data := make([]float32, size.rows*size.cols)
			for i := range data {
				data[i] = float32(i%100) / 10.0
			}
			
			tensor := backend.NewTensor(size.rows, size.cols, data)
			
			// Warm-up
			tensor.Softmax()
			backend.Synchronize()
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				tensor.Softmax()
			}
			backend.Synchronize()
		})
	}
}

// TestMetalKernelPerformanceBaseline establishes baseline metrics for optimization
func TestMetalKernelPerformanceBaseline(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance baseline test in short mode")
	}
	
	backend := NewMetalBackend()
	
	t.Run("MatMul_Baseline", func(t *testing.T) {
		// 512x512 matrix multiplication
		size := 512
		A := backend.NewTensor(size, size, nil)
		B := backend.NewTensor(size, size, nil)
		C := backend.NewTensor(size, size, nil)
		
		aData := make([]float32, size*size)
		bData := make([]float32, size*size)
		for i := range aData {
			aData[i] = 1.0
			bData[i] = 1.0
		}
		A.CopyFromFloat32(aData)
		B.CopyFromFloat32(bData)
		
		// Measure 100 iterations
		iterations := 100
		start := time.Now()
		for i := 0; i < iterations; i++ {
			C.Mul(A, B)
		}
		backend.Synchronize()
		elapsed := time.Since(start)
		
		ops := 2 * int64(size) * int64(size) * int64(size) * int64(iterations)
		gflops := float64(ops) / elapsed.Seconds() / 1e9
		
		t.Logf("MatMul 512x512: %.2f GFLOPS (%.2f ms/iter)", gflops, elapsed.Seconds()*1000/float64(iterations))
		
		// Baseline expectation: >100 GFLOPS on M3 Pro
		if gflops < 50 {
			t.Errorf("MatMul performance below baseline: %.2f GFLOPS (expected >50)", gflops)
		}
	})
	
	t.Run("LayerNorm_Baseline", func(t *testing.T) {
		batch, dim := 256, 384
		data := make([]float32, batch*dim)
		for i := range data {
			data[i] = float32(i) / 100.0
		}
		
		tensor := backend.NewTensor(batch, dim, data)
		gamma := backend.NewTensor(1, dim, nil)
		beta := backend.NewTensor(1, dim, nil)
		
		gammaData := make([]float32, dim)
		betaData := make([]float32, dim)
		for i := range gammaData {
			gammaData[i] = 1.0
		}
		gamma.CopyFromFloat32(gammaData)
		beta.CopyFromFloat32(betaData)
		
		iterations := 1000
		start := time.Now()
		for i := 0; i < iterations; i++ {
			tensor.LayerNorm(gamma, beta, 1e-5)
		}
		backend.Synchronize()
		elapsed := time.Since(start)
		
		t.Logf("LayerNorm %dx%d: %.2f ms/iter", batch, dim, elapsed.Seconds()*1000/float64(iterations))
		
		// Baseline: should be <1ms per iteration
		avgMs := elapsed.Seconds() * 1000 / float64(iterations)
		if avgMs > 2.0 {
			t.Errorf("LayerNorm performance below baseline: %.2f ms/iter (expected <2ms)", avgMs)
		}
	})
}

// TestMetalKernelCorrectness validates kernel output correctness
func TestMetalKernelCorrectness(t *testing.T) {
	backend := NewMetalBackend()
	
	t.Run("MatMul_Correctness", func(t *testing.T) {
		// Small matrix for easy verification
		A := backend.NewTensor(2, 3, []float32{1, 2, 3, 4, 5, 6})
		B := backend.NewTensor(3, 2, []float32{7, 8, 9, 10, 11, 12})
		C := backend.NewTensor(2, 2, nil)
		
		C.Mul(A, B)
		out := C.ToHost()
		
		// Expected: [58, 64, 139, 154]
		expected := []float32{58, 64, 139, 154}
		for i, v := range out {
			if v != expected[i] {
				t.Errorf("MatMul correctness failed at %d: got %f, want %f", i, v, expected[i])
			}
		}
	})
	
	t.Run("LayerNorm_Correctness", func(t *testing.T) {
		// Single row for easy verification
		data := []float32{1, 2, 3, 4}
		tensor := backend.NewTensor(1, 4, data)
		gamma := backend.NewTensor(1, 4, []float32{1, 1, 1, 1})
		beta := backend.NewTensor(1, 4, []float32{0, 0, 0, 0})
		
		tensor.LayerNorm(gamma, beta, 1e-5)
		out := tensor.ToHost()
		
		// Mean should be ~0, std should be ~1
		mean := (out[0] + out[1] + out[2] + out[3]) / 4
		if mean > 0.01 || mean < -0.01 {
			t.Errorf("LayerNorm mean not zero: %f", mean)
		}
	})
}
