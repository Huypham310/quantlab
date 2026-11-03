// Package feed reads live trades from the Coinbase Exchange public websocket.
package feed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/coder/websocket"
)

const DefaultURL = "wss://ws-feed.exchange.coinbase.com"

// Trade is one executed trade, already normalized (prices as numbers, UTC time).
type Trade struct {
	Exchange string
	Symbol   string
	TradeID  int64
	Price    float64
	Size     float64
	Side     string // taker side: "buy" or "sell"
	Time     time.Time
}

// raw "match" message from the "matches" channel
type matchMsg struct {
	Type      string    `json:"type"`
	TradeID   int64     `json:"trade_id"`
	ProductID string    `json:"product_id"`
	Price     string    `json:"price"`
	Size      string    `json:"size"`
	Side      string    `json:"side"`
	Time      time.Time `json:"time"`
	Message   string    `json:"message"`
	Reason    string    `json:"reason"`
}

// ErrSkip means the message is valid but not a trade (subscription ack, heartbeat...).
var ErrSkip = errors.New("not a trade")

// ParseMessage turns one websocket frame into a Trade.
func ParseMessage(data []byte) (Trade, error) {
	var m matchMsg
	if err := json.Unmarshal(data, &m); err != nil {
		return Trade{}, fmt.Errorf("bad json: %w", err)
	}
	switch m.Type {
	case "match", "last_match":
	case "error":
		return Trade{}, fmt.Errorf("coinbase error: %s %s", m.Message, m.Reason)
	default:
		return Trade{}, ErrSkip
	}
	price, err := strconv.ParseFloat(m.Price, 64)
	if err != nil {
		return Trade{}, fmt.Errorf("bad price %q: %w", m.Price, err)
	}
	size, err := strconv.ParseFloat(m.Size, 64)
	if err != nil {
		return Trade{}, fmt.Errorf("bad size %q: %w", m.Size, err)
	}
	// Coinbase gửi side theo maker, đảo lại để thành side của taker (bên chủ động khớp lệnh)
	taker := "buy"
	if m.Side == "buy" {
		taker = "sell"
	}
	return Trade{
		Exchange: "coinbase",
		Symbol:   m.ProductID,
		TradeID:  m.TradeID,
		Price:    price,
		Size:     size,
		Side:     taker,
		Time:     m.Time.UTC(),
	}, nil
}

// Stream connects, subscribes to "matches" for the products, and sends trades to out until
// ctx is cancelled. Dropped connections are retried with exponential backoff.
func Stream(ctx context.Context, url string, products []string, out chan<- Trade) error {
	backoff := NewBackoff(time.Second, 30*time.Second)
	for {
		err := streamOnce(ctx, url, products, out, backoff)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		wait := backoff.Next()
		log.Printf("feed: disconnected (%v), reconnecting in %s", err, wait.Round(time.Millisecond))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
}

func streamOnce(ctx context.Context, url string, products []string, out chan<- Trade, backoff *Backoff) error {
	dialCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(dialCtx, url, nil)
	if err != nil {
		return err
	}
	defer conn.CloseNow()
	conn.SetReadLimit(1 << 20)

	sub, _ := json.Marshal(map[string]any{
		"type":        "subscribe",
		"product_ids": products,
		"channels":    []string{"matches"},
	})
	if err := conn.Write(ctx, websocket.MessageText, sub); err != nil {
		return err
	}
	log.Printf("feed: subscribed to %v", products)

	gotTrade := false
	for {
		// no trade for a minute on BTC means the socket is silently dead
		readCtx, cancelRead := context.WithTimeout(ctx, time.Minute)
		_, data, err := conn.Read(readCtx)
		cancelRead()
		if err != nil {
			return err
		}
		trade, err := ParseMessage(data)
		if errors.Is(err, ErrSkip) {
			continue
		}
		if err != nil {
			return err
		}
		if !gotTrade {
			// only reset the backoff once data actually flows again
			backoff.Reset()
			gotTrade = true
		}
		select {
		case out <- trade:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
