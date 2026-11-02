package feed

import (
	"errors"
	"testing"
	"time"
)

func TestParseMatch(t *testing.T) {
	msg := `{"type":"match","trade_id":865127782,"maker_order_id":"a","taker_order_id":"b",
		"side":"buy","size":"0.00513192","price":"64321.5","product_id":"BTC-USD",
		"sequence":1,"time":"2026-11-01T18:04:05.123456Z"}`
	tr, err := ParseMessage([]byte(msg))
	if err != nil {
		t.Fatal(err)
	}
	want := Trade{
		Exchange: "coinbase",
		Symbol:   "BTC-USD",
		TradeID:  865127782,
		Price:    64321.5,
		Size:     0.00513192,
		Side:     "sell", // maker bought -> the taker sold
		Time:     time.Date(2026, 11, 1, 18, 4, 5, 123456000, time.UTC),
	}
	if tr != want {
		t.Fatalf("got %+v\nwant %+v", tr, want)
	}
}

func TestParseFlipsMakerSide(t *testing.T) {
	tr, err := ParseMessage([]byte(`{"type":"last_match","trade_id":1,"side":"sell","size":"1","price":"2",
		"product_id":"ETH-USD","time":"2026-11-01T00:00:00Z"}`))
	if err != nil {
		t.Fatal(err)
	}
	if tr.Side != "buy" {
		t.Fatalf("maker sell should become taker buy, got %q", tr.Side)
	}
}

func TestParseSkipsNonTrades(t *testing.T) {
	for _, msg := range []string{
		`{"type":"subscriptions","channels":[{"name":"matches","product_ids":["BTC-USD"]}]}`,
		`{"type":"heartbeat","sequence":5}`,
	} {
		if _, err := ParseMessage([]byte(msg)); !errors.Is(err, ErrSkip) {
			t.Errorf("%s: want ErrSkip, got %v", msg, err)
		}
	}
}

func TestParseErrors(t *testing.T) {
	cases := map[string]string{
		"not json":    `{oops`,
		"error frame": `{"type":"error","message":"Failed to subscribe","reason":"BAD-USD is not a valid product"}`,
		"bad price":   `{"type":"match","price":"abc","size":"1","time":"2026-11-01T00:00:00Z"}`,
		"bad size":    `{"type":"match","price":"1","size":"","time":"2026-11-01T00:00:00Z"}`,
	}
	for name, msg := range cases {
		_, err := ParseMessage([]byte(msg))
		if err == nil || errors.Is(err, ErrSkip) {
			t.Errorf("%s: want a real error, got %v", name, err)
		}
	}
}
