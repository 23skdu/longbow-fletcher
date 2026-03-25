package embeddings

import (
	"sync"
)

type LoRAConfig struct {
	Rank          int
	Alpha         int
	Dropout       float32
	TargetModules []string
}

type LoRAParameters struct {
	A     map[string][][]float32
	B     map[string][][]float32
	Scale float32
}

type LoRAResult struct {
	AppliedModules []string
	Output         []float32
}

type PrefixCache struct {
	mu    sync.RWMutex
	cache map[string]*PrefixCacheEntry
	hits  int64
	miss  int64
}

type PrefixCacheEntry struct {
	Key        string
	TokenIDs   []int
	Embeddings []float32
	Length     int
	Hits       int64
}

func NewPrefixCache() *PrefixCache {
	return &PrefixCache{
		cache: make(map[string]*PrefixCacheEntry),
	}
}

func (p *PrefixCache) Get(tokenIDs []int) ([]float32, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	key := cacheKey(tokenIDs)
	if entry, ok := p.cache[key]; ok {
		p.hits++
		entry.Hits++
		return entry.Embeddings, true
	}
	p.miss++
	return nil, false
}

func (p *PrefixCache) Put(tokenIDs []int, embeddings []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := cacheKey(tokenIDs)
	p.cache[key] = &PrefixCacheEntry{
		Key:        key,
		TokenIDs:   tokenIDs,
		Embeddings: embeddings,
		Length:     len(tokenIDs),
	}
}

func (p *PrefixCache) Stats() (hits, miss int64, ratio float64) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	total := p.hits + p.miss
	if total > 0 {
		ratio = float64(p.hits) / float64(total)
	}
	return p.hits, p.miss, ratio
}

func (p *PrefixCache) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cache = make(map[string]*PrefixCacheEntry)
	p.hits = 0
	p.miss = 0
}

func cacheKey(tokenIDs []int) string {
	if len(tokenIDs) > 10 {
		tokenIDs = tokenIDs[:10]
	}
	key := 0
	for _, id := range tokenIDs {
		key = key*31 + id
	}
	return string(rune(key))
}

type ChunkedPrefill struct {
	MaxChunkSize int
	Overlap      int
}

func NewChunkedPrefill(maxChunkSize, overlap int) *ChunkedPrefill {
	return &ChunkedPrefill{
		MaxChunkSize: maxChunkSize,
		Overlap:      overlap,
	}
}

func (c *ChunkedPrefill) ChunkTokens(tokenIDs []int) [][]int {
	if len(tokenIDs) <= c.MaxChunkSize {
		return [][]int{tokenIDs}
	}

	var chunks [][]int
	for i := 0; i < len(tokenIDs); i += c.MaxChunkSize - c.Overlap {
		end := i + c.MaxChunkSize
		if end > len(tokenIDs) {
			end = len(tokenIDs)
		}
		chunks = append(chunks, tokenIDs[i:end])
		if end == len(tokenIDs) {
			break
		}
	}
	return chunks
}

type FP8KVCache struct {
	Enabled   bool
	Scale     float32
	QuantBits int
}

func NewFP8KVCache(enabled bool) *FP8KVCache {
	return &FP8KVCache{
		Enabled:   enabled,
		Scale:     1.0,
		QuantBits: 8,
	}
}

func (f *FP8KVCache) Quantize(data []float32) ([]byte, float32) {
	if !f.Enabled {
		return nil, 1.0
	}

	minVal := float32(1e9)
	maxVal := float32(-1e9)
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	scale := (maxVal - minVal) / 255.0
	if scale == 0 {
		scale = 1.0
	}

	quantized := make([]byte, len(data))
	for i, v := range data {
		val := int((v - minVal) / scale)
		if val > 255 {
			val = 255
		}
		if val < 0 {
			val = 0
		}
		quantized[i] = byte(val)
	}

	f.Scale = scale
	return quantized, scale
}

func (f *FP8KVCache) Dequantize(data []byte, scale float32) []float32 {
	if !f.Enabled || len(data) == 0 {
		return nil
	}

	result := make([]float32, len(data))
	for i, v := range data {
		result[i] = float32(v) * scale
	}
	return result
}

type StructuredOutput struct {
	Schema    map[string]interface{}
	Validator func(interface{}) bool
}

func NewStructuredOutput(schema map[string]interface{}) *StructuredOutput {
	return &StructuredOutput{
		Schema: schema,
	}
}

func (s *StructuredOutput) Validate(output interface{}) bool {
	if s.Validator != nil {
		return s.Validator(output)
	}
	return true
}
