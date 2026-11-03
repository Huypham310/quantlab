package feed

import "testing"

// a real match frame as Coinbase sends it
var benchFrame = []byte(`{"type":"match","trade_id":865127782,"maker_order_id":"a","taker_order_id":"b",` +
	`"side":"buy","size":"0.00513192","price":"64321.5","product_id":"BTC-USD","sequence":1,` +
	`"time":"2026-11-01T18:04:05.123456Z"}`)

// go test ./internal/feed -bench Parse -benchmem
func BenchmarkParseMessage(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ParseMessage(benchFrame); err != nil {
			b.Fatal(err)
		}
	}
}
