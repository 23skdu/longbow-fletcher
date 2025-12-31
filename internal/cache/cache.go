package cache

import (
	"sync"
)

// VectorCache defines a generic interface for caching embeddings.
type VectorCache interface {
	// Get retrieves a vector from the cache.
	Get(key string) ([]float32, bool)
	// Put stores a vector in the cache.
	Put(key string, vec []float32)
	// Size returns the number of items in the cache.
	Size() int
}

// MapCache is a simple in-memory implementation of VectorCache.
type MapCache struct {
	data map[string][]float32
	mu   sync.RWMutex
}

func NewMapCache() *MapCache {
	return &MapCache{
		data: make(map[string][]float32),
	}
}

func (c *MapCache) Get(key string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	// Return copy to avoid modification of cached value
	if v, ok := c.data[key]; ok {
		dst := make([]float32, len(v))
		copy(dst, v)
		return dst, true
	}
	return nil, false
}

func (c *MapCache) Put(key string, vec []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Store copy
	dst := make([]float32, len(vec))
	copy(dst, vec)
	c.data[key] = dst
}

func (c *MapCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.data)
}
