package feed

import (
	"math/rand/v2"
	"time"
)

// Backoff doubles the wait after each failure up to max, with +-20% jitter so many
// reconnecting clients don't hit the exchange at the same instant.
type Backoff struct {
	base, max, current time.Duration
}

func NewBackoff(base, max time.Duration) *Backoff {
	return &Backoff{base: base, max: max, current: base}
}

// Next returns how long to wait before the next attempt.
func (b *Backoff) Next() time.Duration {
	wait := b.current
	b.current = min(b.current*2, b.max)
	jitter := 0.8 + rand.Float64()*0.4
	return time.Duration(float64(wait) * jitter)
}

// Reset goes back to the base delay after a healthy connection.
func (b *Backoff) Reset() { b.current = b.base }
