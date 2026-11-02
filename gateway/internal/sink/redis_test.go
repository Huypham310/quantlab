package sink

import (
	"context"
	"testing"
	"time"

	"github.com/Huypham310/quantlab/gateway/internal/feed"
	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

var trade = feed.Trade{
	Exchange: "coinbase",
	Symbol:   "BTC-USD",
	TradeID:  42,
	Price:    64321.5,
	Size:     0.001,
	Side:     "buy",
	Time:     time.Date(2026, 11, 1, 18, 4, 5, 123_000_000, time.UTC),
}

func TestFields(t *testing.T) {
	f := Fields(trade)
	want := map[string]any{
		"exchange": "coinbase", "symbol": "BTC-USD", "trade_id": "42",
		"price": "64321.5", "size": "0.001", "side": "buy", "ts_ms": "1793556245123",
	}
	for k, v := range want {
		if f[k] != v {
			t.Errorf("%s: got %v want %v", k, f[k], v)
		}
	}
	if len(f) != len(want) {
		t.Errorf("unexpected extra fields: %v", f)
	}
}

func TestWriteAppendsToStream(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()
	w := &Writer{Client: client, Stream: "trades", MaxLen: 1000}
	ctx := context.Background()

	for range 3 {
		if err := w.Write(ctx, trade); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := client.XRange(ctx, "trades", "-", "+").Result()
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("want 3 entries, got %d", len(entries))
	}
	if entries[0].Values["price"] != "64321.5" || entries[0].Values["symbol"] != "BTC-USD" {
		t.Fatalf("unexpected entry %v", entries[0].Values)
	}
}

func TestWriteCapsStreamLength(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()
	w := &Writer{Client: client, Stream: "trades", MaxLen: 10}
	ctx := context.Background()

	for range 50 {
		if err := w.Write(ctx, trade); err != nil {
			t.Fatal(err)
		}
	}
	n := client.XLen(ctx, "trades").Val()
	// "~" trimming may keep a few extra entries on real redis, but never grows unbounded
	if n > 20 {
		t.Fatalf("stream not trimmed: %d entries", n)
	}
}

func TestWriteFailsWhenRedisIsDown(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr(), MaxRetries: -1})
	defer client.Close()
	mr.Close()
	w := &Writer{Client: client, Stream: "trades", MaxLen: 10}
	if err := w.Write(context.Background(), trade); err == nil {
		t.Fatal("want an error when redis is gone")
	}
}
