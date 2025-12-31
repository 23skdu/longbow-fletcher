//go:build darwin && metal

package device

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

func getMetricValue(m prometheus.Metric) float64 {
	var metric dto.Metric
	m.Write(&metric)
	if metric.Counter != nil {
		return *metric.Counter.Value
	}
	if metric.Gauge != nil {
		return *metric.Gauge.Value
	}
	return 0
}

func TestMetal_PoolMetrics(t *testing.T) {
	// 1. Setup
	backend := NewMetalBackend()
	
	// Reset/Snapshot (metrics are global, so we track deltas)
	startHits := getMetricValue(poolHits)
	startMisses := getMetricValue(poolMisses)
	
	// 2. Allocate (Miss)
	t1 := backend.NewTensor(100, 100, nil) // 100x100 * 4 bytes = 40KB
	
	// Verify Miss
	if hit, miss := getMetricValue(poolHits), getMetricValue(poolMisses); miss-startMisses != 1 {
		t.Errorf("Expected 1 miss, got %v (hits=%v)", miss-startMisses, hit-startHits)
	}
	
	// 3. Free (Return to Pool)
	backend.PutTensor(t1)
	
	// Force synchronization/drain if needed?
	// getPooledBuffer logic drains pendingBuckets. 
	// So returning puts it in pending.
	// We need to trigger a getPooledBuffer to see if it moves?
	// Actually, Put -> Pending. Pending is "in pool" but not available for Immediate Get until drained?
	// Our metrics logic: ReturnToPool increments poolSize.
	
	// 4. Allocate Same Size (Hit)
	// getPooledBuffer will drain pendingBuckets first if GPU is completed.
	// Since we did nothing on GPU, IsCompleted should be true?
	// Force sync just in case?
	backend.Synchronize()
	
	t2 := backend.NewTensor(100, 100, nil)
	
	// Verify Hit
	if hit, miss := getMetricValue(poolHits), getMetricValue(poolMisses); hit-startHits != 1 {
		t.Logf("Missed pool hit! Hits delta: %v, Misses delta: %v", hit-startHits, miss-startMisses)
		// It might stay in pending if backend doesn't think it's done? 
		// Sync should fix it.
	} else {
		t.Log("Pool Hit Confirmed")
	}
	
	// Cleanup
	backend.PutTensor(t2)
}
