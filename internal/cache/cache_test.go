package cache

import (
	"testing"
)

func TestMapCache(t *testing.T) {
	c := NewMapCache()

	if c.Size() != 0 {
		t.Errorf("New cache should have size 0, got %d", c.Size())
	}

	vec := []float32{1.0, 2.0, 3.0}
	c.Put("key1", vec)

	if c.Size() != 1 {
		t.Errorf("After put, size should be 1, got %d", c.Size())
	}

	retrieved, ok := c.Get("key1")
	if !ok {
		t.Error("Expected to find key1")
	}
	if len(retrieved) != len(vec) {
		t.Errorf("Retrieved length mismatch: got %d, want %d", len(retrieved), len(vec))
	}
	for i := range retrieved {
		if retrieved[i] != vec[i] {
			t.Errorf("Retrieved value mismatch at index %d: got %f, want %f", i, retrieved[i], vec[i])
		}
	}
}

func TestMapCache_MissingKey(t *testing.T) {
	c := NewMapCache()

	_, ok := c.Get("nonexistent")
	if ok {
		t.Error("Expected false for missing key")
	}
}

func TestMapCache_Overwrite(t *testing.T) {
	c := NewMapCache()

	c.Put("key1", []float32{1.0, 2.0})
	c.Put("key1", []float32{3.0, 4.0, 5.0})

	retrieved, _ := c.Get("key1")
	if len(retrieved) != 3 {
		t.Errorf("Expected length 3 after overwrite, got %d", len(retrieved))
	}
	if retrieved[0] != 3.0 {
		t.Errorf("Expected 3.0, got %f", retrieved[0])
	}
}

func TestMapCache_Concurrent(t *testing.T) {
	c := NewMapCache()

	done := make(chan bool)
	go func() {
		for i := 0; i < 100; i++ {
			c.Put("key", []float32{float32(i)})
		}
		done <- true
	}()

	for i := 0; i < 100; i++ {
		c.Get("key")
	}
	<-done
}
