package feed

import (
	"testing"
	"time"
)

func TestBackoffDoublesUpToMax(t *testing.T) {
	b := NewBackoff(time.Second, 8*time.Second)
	// expected base delays before jitter
	for i, want := range []time.Duration{1, 2, 4, 8, 8, 8} {
		got := b.Next()
		lo, hi := time.Duration(float64(want*time.Second)*0.8), time.Duration(float64(want*time.Second)*1.2)
		if got < lo || got > hi {
			t.Fatalf("attempt %d: %s not within [%s, %s]", i, got, lo, hi)
		}
	}
}

func TestBackoffReset(t *testing.T) {
	b := NewBackoff(time.Second, time.Minute)
	for range 5 {
		b.Next()
	}
	b.Reset()
	if got := b.Next(); got > 1200*time.Millisecond {
		t.Fatalf("after reset want ~1s, got %s", got)
	}
}
