package model

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// LayerDuration tracks time spent in specific model layers
	LayerDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "fletcher_gpu_layer_duration_seconds",
		Help:    "Time spent in specific model layers",
		Buckets: []float64{0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1},
	}, []string{"layer_type", "device"})
)
