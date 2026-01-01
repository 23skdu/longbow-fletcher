package client

import (
	"sync"
	"time"
)

// State represents the state of the circuit breaker.
type State int

const (
	StateClosed State = iota
	StateOpen
	StateHalfOpen
)

// CircuitBreaker implements a standard circuit breaker pattern.
// It is thread-safe.
type CircuitBreaker struct {
	mu           sync.Mutex
	state        State
	failures     int
	maxFailures  int
	timeout      time.Duration
	lastFailure  time.Time
}

// NewCircuitBreaker creates a new CircuitBreaker.
// maxFailures: Number of consecutive failures before opening the circuit.
// timeout: Duration to wait before attempting to half-open the circuit.
func NewCircuitBreaker(maxFailures int, timeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		state:       StateClosed,
		maxFailures: maxFailures,
		timeout:     timeout,
	}
}

// Allow checks if a request is allowed to proceed.
// It returns true if the circuit is Closed or Half-Open.
// It limits traffic in restricted states (though this simple impl allows single probes).
func (cb *CircuitBreaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == StateClosed {
		return true
	}

	if cb.state == StateOpen {
		if time.Since(cb.lastFailure) > cb.timeout {
			// Transition to Half-Open to probe
			cb.state = StateHalfOpen
			return true
		}
		return false
	}

	// StateHalfOpen
	// We allow one request to probe. 
	// In a real high-concurrency scenario, we might want to strictly limit this to 1 concurrent request.
	// For this use case (single loop), it usually works naturally.
	return true
}

// Success records a successful operation.
func (cb *CircuitBreaker) Success() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == StateHalfOpen {
		cb.state = StateClosed
		cb.failures = 0
	} else if cb.state == StateClosed {
		cb.failures = 0
	}
}

// Failure records a failed operation.
func (cb *CircuitBreaker) Failure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failures++
	cb.lastFailure = time.Now()
	
	if cb.state == StateClosed {
		if cb.failures >= cb.maxFailures {
			cb.state = StateOpen
		}
	} else if cb.state == StateHalfOpen {
		cb.state = StateOpen
	}
}

// State returns the current state.
func (cb *CircuitBreaker) State() State {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.state
}
