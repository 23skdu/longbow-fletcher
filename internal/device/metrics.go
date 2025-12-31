//go:build metal

package device

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	poolHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fletcher_metal_pool_hits_total",
		Help: "Total number of successful buffer pool retrievals",
	})

	poolMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "fletcher_metal_pool_misses_total",
		Help: "Total number of buffer pool misses (allocations)",
	})

	poolSizeBytes = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fletcher_metal_pool_size_bytes",
		Help: "Current total size of buffers in the pool in bytes",
	})
	
	poolBuffers = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fletcher_metal_pool_buffers_count",
		Help: "Current total number of buffers in the pool",
	})
)
