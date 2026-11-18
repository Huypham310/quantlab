package metrics

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

// scrape runs one GET /metrics against the handler and returns the body.
func scrape(t *testing.T, g *Gateway) string {
	t.Helper()
	rec := httptest.NewRecorder()
	g.Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	body, _ := io.ReadAll(rec.Body)
	return string(body)
}

func TestMetricNamesAndValues(t *testing.T) {
	g := New()
	g.TradesWritten.Add(3)
	g.TradesFailed.Inc()
	g.Reconnects.Inc()
	g.LastTradeTime.SetToCurrentTime()
	g.WriteLatency.Observe(0.0004)
	g.WatchBacklog(func() int { return 0 })

	out := scrape(t, g)
	for _, want := range []string{
		"quantlab_gateway_trades_written_total 3",
		"quantlab_gateway_trades_failed_total 1",
		"quantlab_gateway_reconnects_total 1",
		"quantlab_gateway_last_trade_timestamp_seconds ",
		"quantlab_gateway_redis_write_seconds_count 1",
		"quantlab_gateway_channel_backlog 0",
		"go_goroutines", // runtime collector is on too
	} {
		if !strings.Contains(out, want) {
			t.Errorf("metrics output is missing %q", want)
		}
	}
}

func TestServeEmptyAddrIsNoop(t *testing.T) {
	// "" means disabled; must return immediately without opening a port
	New().Serve(t.Context(), "")
}
