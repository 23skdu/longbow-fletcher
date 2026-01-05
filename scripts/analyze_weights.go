//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/23skdu/longbow-fletcher/internal/device"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/weights"
)

func main() {
	// Create FP32 backend to load weights
	backend := device.NewMetalBackend()
	config := model.DefaultBertTinyConfig()
	bert := model.NewBertModelWithBackend(config, backend)

	// Load weights
	loader := weights.NewLoader(bert)
	if err := loader.LoadFromSafeTensors("bert_tiny.safetensors"); err != nil {
		log.Fatalf("Failed to load weights: %v", err)
	}

	// Analyze all weights
	stats := make(map[string]WeightStats)

	// Embeddings
	stats["word_embeddings"] = analyzeWeight(bert.Embeddings.WordEmbeddings)
	stats["position_embeddings"] = analyzeWeight(bert.Embeddings.PositionEmbeddings)
	stats["token_type_embeddings"] = analyzeWeight(bert.Embeddings.TokenTypeEmbeddings)
	stats["embedding_ln_gamma"] = analyzeWeight(bert.Embeddings.LayerNorm.Gamma)
	stats["embedding_ln_beta"] = analyzeWeight(bert.Embeddings.LayerNorm.Beta)

	// Encoder layers
	for i, layer := range bert.Encoder.Layers {
		prefix := fmt.Sprintf("layer_%d", i)
		stats[prefix+"_query"] = analyzeWeight(layer.Attention.Self.Query)
		stats[prefix+"_query_bias"] = analyzeWeight(layer.Attention.Self.QueryBias)
		stats[prefix+"_key"] = analyzeWeight(layer.Attention.Self.Key)
		stats[prefix+"_key_bias"] = analyzeWeight(layer.Attention.Self.KeyBias)
		stats[prefix+"_value"] = analyzeWeight(layer.Attention.Self.Value)
		stats[prefix+"_value_bias"] = analyzeWeight(layer.Attention.Self.ValueBias)
		stats[prefix+"_attn_out"] = analyzeWeight(layer.Attention.Output.Dense)
		stats[prefix+"_attn_out_bias"] = analyzeWeight(layer.Attention.Output.Bias)
		stats[prefix+"_attn_ln_gamma"] = analyzeWeight(layer.Attention.Output.LayerNorm.Gamma)
		stats[prefix+"_attn_ln_beta"] = analyzeWeight(layer.Attention.Output.LayerNorm.Beta)
		stats[prefix+"_intermediate"] = analyzeWeight(layer.Intermediate.Dense)
		stats[prefix+"_intermediate_bias"] = analyzeWeight(layer.Intermediate.Bias)
		stats[prefix+"_output"] = analyzeWeight(layer.Output.Dense)
		stats[prefix+"_output_bias"] = analyzeWeight(layer.Output.Bias)
		stats[prefix+"_output_ln_gamma"] = analyzeWeight(layer.Output.LayerNorm.Gamma)
		stats[prefix+"_output_ln_beta"] = analyzeWeight(layer.Output.LayerNorm.Beta)
	}

	// Pooler
	stats["pooler"] = analyzeWeight(bert.Pooler.Dense)
	stats["pooler_bias"] = analyzeWeight(bert.Pooler.Bias)

	// Output results
	output := map[string]interface{}{
		"summary": summarize(stats),
		"details": stats,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(output); err != nil {
		log.Fatalf("Failed to encode output: %v", err)
	}
}

type WeightStats struct {
	Min             float32 `json:"min"`
	Max             float32 `json:"max"`
	Mean            float32 `json:"mean"`
	AbsMax          float32 `json:"abs_max"`
	OutOfRangeCount int     `json:"out_of_range_count"`
	OutOfRangeRatio float64 `json:"out_of_range_ratio"`
	NaNCount        int     `json:"nan_count"`
	InfCount        int     `json:"inf_count"`
	TotalElements   int     `json:"total_elements"`
}

func analyzeWeight(t device.Tensor) WeightStats {
	data := t.ToHost()

	stats := WeightStats{
		Min:           math.MaxFloat32,
		Max:           -math.MaxFloat32,
		TotalElements: len(data),
	}

	const maxFP16 = 65504.0
	const minNormalFP16 = 6.10351562e-5

	sum := float64(0)
	for _, v := range data {
		// Check for special values
		if math.IsNaN(float64(v)) {
			stats.NaNCount++
			continue
		}
		if math.IsInf(float64(v), 0) {
			stats.InfCount++
			continue
		}

		// Min/Max
		if v < stats.Min {
			stats.Min = v
		}
		if v > stats.Max {
			stats.Max = v
		}

		// Abs Max
		absV := v
		if absV < 0 {
			absV = -absV
		}
		if absV > stats.AbsMax {
			stats.AbsMax = absV
		}

		// Check FP16 range
		if absV > maxFP16 || (absV < minNormalFP16 && absV > 0) {
			stats.OutOfRangeCount++
		}

		sum += float64(v)
	}

	if stats.TotalElements > 0 {
		stats.Mean = float32(sum / float64(stats.TotalElements))
		stats.OutOfRangeRatio = float64(stats.OutOfRangeCount) / float64(stats.TotalElements)
	}

	return stats
}

func summarize(stats map[string]WeightStats) map[string]interface{} {
	totalOutOfRange := 0
	totalElements := 0
	totalNaN := 0
	totalInf := 0
	problematicWeights := []string{}

	for name, s := range stats {
		totalOutOfRange += s.OutOfRangeCount
		totalElements += s.TotalElements
		totalNaN += s.NaNCount
		totalInf += s.InfCount

		if s.OutOfRangeRatio > 0.01 || s.NaNCount > 0 || s.InfCount > 0 {
			problematicWeights = append(problematicWeights, name)
		}
	}

	return map[string]interface{}{
		"total_elements":      totalElements,
		"total_out_of_range":  totalOutOfRange,
		"out_of_range_ratio":  float64(totalOutOfRange) / float64(totalElements),
		"total_nan":           totalNaN,
		"total_inf":           totalInf,
		"problematic_weights": problematicWeights,
	}
}
