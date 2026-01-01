package client

import (
	"testing"
	"time"
)

func TestCircuitBreaker(t *testing.T) {
	// Configure for fast testing: 3 failures, 100ms timeout
	cb := NewCircuitBreaker(3, 100*time.Millisecond)

	// Initial State: Closed
	if cb.State() != StateClosed {
		t.Errorf("Expected Closed state, got %v", cb.State())
	}
	if !cb.Allow() {
		t.Error("Should allow requests in Closed state")
	}

	// Trigger Failures
	cb.Failure()
	cb.Failure()
	if cb.State() != StateClosed {
		t.Errorf("Should remain Closed after 2 failures")
	}

	cb.Failure()
	// Should trip now (3 failures)
	if cb.State() != StateOpen {
		t.Errorf("Expected Open state after 3 failures")
	}
	if cb.Allow() {
		t.Error("Should NOT allow requests in Open state")
	}

	// Wait for timeout (Half-Open)
	time.Sleep(150 * time.Millisecond)
	
	// First call should probe (allow) and transit to Half-Open internal check
	if !cb.Allow() {
		t.Error("Should allow probe request after timeout")
	}
	
	// Check state transition
	if cb.State() != StateHalfOpen {
		t.Errorf("Expected HalfOpen state, got %v", cb.State())
	}

	// Case A: Probe Fails -> Open again
	cb.Failure()
	if cb.State() != StateOpen {
		t.Errorf("Expected Open state after probe failure")
	}

	// Wait again
	time.Sleep(150 * time.Millisecond)
	cb.Allow() // Trigger Half-Open
	
	// Case B: Probe Succeeds -> Closed
	cb.Success()
	if cb.State() != StateClosed {
		t.Errorf("Expected Closed state after probe success")
	}
	if cb.failures != 0 {
		t.Errorf("Failures should be reset")
	}
}
