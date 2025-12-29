package embeddings

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// GPU performance metrics
	gpuThroughput = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "fletcher_gpu_throughput",
		Help: "GPU throughput in sequences per second",
	}, []string{"device"})

	gpuBatchTime = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "fletcher_gpu_batch_time_seconds",
		Help: "Last batch processing time in seconds",
	}, []string{"device"})

	gpuBatchCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "fletcher_gpu_batch_count_total",
		Help: "Total number of batches processed by GPU",
	}, []string{"device"})

	gpuSequencesProcessed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "fletcher_gpu_sequences_total",
		Help: "Total number of sequences processed by GPU",
	}, []string{"device"})

	gpuTokensProcessed = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "fletcher_gpu_tokens_total",
		Help: "Total number of tokens processed by GPU",
	}, []string{"device"})

	gpuWeight = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "fletcher_gpu_weight",
		Help: "Current load balancing weight for GPU (based on throughput)",
	}, []string{"device"})

	// TODO: Implement load balance efficiency metric
	// loadBalanceEfficiency = promauto.NewGauge(prometheus.GaugeOpts{
	// 	Name: "fletcher_load_balance_efficiency",
	// 	Help: "Load balancing efficiency (0-1, 1 = perfect balance)",
	// })

	// Tokenization metrics
	tokenizationDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "fletcher_tokenization_duration_seconds",
		Help:    "Time spent in tokenization",
		Buckets: prometheus.DefBuckets,
	})

	tokensPerSecond = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "fletcher_tokenization_throughput",
		Help: "Tokenization throughput in tokens/second",
	})
)
