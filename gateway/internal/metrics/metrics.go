// Package metrics holds the gateway's Prometheus counters and the /metrics HTTP handler.
package metrics

import (
	"context"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Gateway groups every metric the gateway exports. A struct (not package globals) so tests
// can build a fresh one with its own registry.
type Gateway struct {
	Registry      *prometheus.Registry
	TradesWritten prometheus.Counter
	TradesFailed  prometheus.Counter
	Reconnects    prometheus.Counter
	LastTradeTime prometheus.Gauge
	WriteLatency  prometheus.Histogram
}

// New registers all metrics (plus Go runtime and process stats) on a private registry.
func New() *Gateway {
	g := &Gateway{
		Registry: prometheus.NewRegistry(),
		TradesWritten: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "quantlab_gateway_trades_written_total",
			Help: "Trades appended to the Redis stream.",
		}),
		TradesFailed: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "quantlab_gateway_trades_failed_total",
			Help: "Trades dropped because the Redis write failed.",
		}),
		Reconnects: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "quantlab_gateway_reconnects_total",
			Help: "Times the Coinbase websocket dropped and was re-dialled.",
		}),
		LastTradeTime: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "quantlab_gateway_last_trade_timestamp_seconds",
			Help: "Unix time of the last trade written; time() minus this is the feed lag.",
		}),
		WriteLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name: "quantlab_gateway_redis_write_seconds",
			Help: "Latency of one XADD.",
			// redis on localhost answers in well under a millisecond
			Buckets: []float64{0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.25},
		}),
	}
	g.Registry.MustRegister(
		g.TradesWritten, g.TradesFailed, g.Reconnects, g.LastTradeTime, g.WriteLatency,
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)
	return g
}

// WatchBacklog exports how many trades sit between the websocket reader and the Redis writer.
// It is a GaugeFunc, so the value is read at scrape time and can't go stale.
func (g *Gateway) WatchBacklog(depth func() int) {
	g.Registry.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "quantlab_gateway_channel_backlog",
		Help: "Trades waiting between the websocket reader and the Redis writer.",
	}, func() float64 { return float64(depth()) }))
}

// Handler serves the registry in the Prometheus text format.
func (g *Gateway) Handler() http.Handler {
	return promhttp.HandlerFor(g.Registry, promhttp.HandlerOpts{})
}

// Serve exposes /metrics on addr until ctx is cancelled. An empty addr disables it.
func (g *Gateway) Serve(ctx context.Context, addr string) {
	if addr == "" {
		return
	}
	mux := http.NewServeMux()
	mux.Handle("/metrics", g.Handler())
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()
	go func() {
		// a busy port shouldn't kill the gateway: metrics are nice to have, trades are the job
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("metrics server stopped: %v", err)
		}
	}()
	log.Printf("metrics on http://%s/metrics", addr)
}
