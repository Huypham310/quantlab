package feed

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// fakeCoinbase accepts websocket clients, checks the subscribe message, then sends `trades`
// match frames per connection and drops the connection.
func fakeCoinbase(t *testing.T, trades int) (url string, conns *atomic.Int32) {
	t.Helper()
	conns = &atomic.Int32{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer c.CloseNow()
		n := conns.Add(1)
		ctx := r.Context()

		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var sub struct {
			Type       string   `json:"type"`
			ProductIDs []string `json:"product_ids"`
			Channels   []string `json:"channels"`
		}
		if json.Unmarshal(data, &sub) != nil || sub.Type != "subscribe" || sub.Channels[0] != "matches" {
			t.Errorf("unexpected subscribe message: %s", data)
			return
		}
		_ = c.Write(ctx, websocket.MessageText, []byte(`{"type":"subscriptions","channels":[]}`))
		for i := range trades {
			msg := fmt.Sprintf(`{"type":"match","trade_id":%d,"side":"buy","size":"0.1","price":"100.5",`+
				`"product_id":"%s","time":"2026-11-01T00:00:0%dZ"}`, int(n)*100+i, sub.ProductIDs[0], i)
			if err := c.Write(ctx, websocket.MessageText, []byte(msg)); err != nil {
				return
			}
		}
		// simulate the exchange dropping us
		c.Close(websocket.StatusGoingAway, "bye")
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), conns
}

func TestStreamDeliversTradesAndReconnects(t *testing.T) {
	url, conns := fakeCoinbase(t, 2)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	out := make(chan Trade, 16)
	done := make(chan error, 1)
	go func() { done <- Stream(ctx, url, []string{"ETH-USD"}, out) }()

	// 2 trades from the first connection, then 2 more after the reconnect (~1s backoff)
	var got []Trade
	for len(got) < 4 {
		select {
		case tr := <-out:
			got = append(got, tr)
		case <-ctx.Done():
			t.Fatalf("only got %d trades before timeout", len(got))
		}
	}
	cancel()
	if err := <-done; err != context.Canceled {
		t.Fatalf("Stream should stop with context.Canceled, got %v", err)
	}

	if conns.Load() < 2 {
		t.Fatalf("expected a reconnect, saw %d connection(s)", conns.Load())
	}
	if got[0].Symbol != "ETH-USD" || got[0].TradeID != 100 || got[2].TradeID != 200 {
		t.Fatalf("unexpected trades: %+v", got)
	}
}

func TestStreamStopsOnCancelWhileWaitingToReconnect(t *testing.T) {
	// nothing listens here, so every dial fails and Stream sits in its backoff wait
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- Stream(ctx, "ws://127.0.0.1:1", []string{"BTC-USD"}, make(chan Trade)) }()

	time.Sleep(200 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Stream didn't return after cancel")
	}
}
