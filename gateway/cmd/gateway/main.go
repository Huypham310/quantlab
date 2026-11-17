// Command gateway streams live trades from the Coinbase public feed into Redis Streams.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/Huypham310/quantlab/gateway/internal/feed"
	"github.com/Huypham310/quantlab/gateway/internal/metrics"
	"github.com/Huypham310/quantlab/gateway/internal/sink"
	"github.com/redis/go-redis/v9"
)

func main() {
	products := flag.String("products", "BTC-USD,ETH-USD", "comma-separated Coinbase product ids")
	redisURL := flag.String("redis", envOr("QUANTLAB_REDIS_URL", "redis://localhost:6379/0"), "redis url")
	stream := flag.String("stream", envOr("QUANTLAB_TRADES_STREAM", "trades"), "redis stream name")
	maxLen := flag.Int64("maxlen", 500_000, "approximate cap on stream length")
	// 9100 so it doesn't clash with the API (8000) / Prometheus (9090); "" turns metrics off
	metricsAddr := flag.String("metrics-addr", envOr("QUANTLAB_GATEWAY_METRICS_ADDR", ":9100"),
		"listen address for Prometheus /metrics (empty to disable)")
	flag.Parse()

	opts, err := redis.ParseURL(*redisURL)
	if err != nil {
		log.Fatalf("bad redis url: %v", err)
	}
	client := redis.NewClient(opts)
	defer client.Close()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := client.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis not reachable at %s: %v", *redisURL, err)
	}

	m := metrics.New()
	m.Serve(ctx, *metricsAddr)

	writer := &sink.Writer{Client: client, Stream: *stream, MaxLen: *maxLen}
	// buffered so a slow redis write doesn't stall reading the socket right away
	trades := make(chan feed.Trade, 1024)
	m.WatchBacklog(func() int { return len(trades) })
	var written, failed atomic.Int64

	go func() {
		for t := range trades {
			start := time.Now()
			if err := writer.Write(ctx, t); err != nil {
				failed.Add(1)
				m.TradesFailed.Inc()
				if ctx.Err() == nil {
					log.Printf("redis write failed: %v", err)
				}
				continue
			}
			m.WriteLatency.Observe(time.Since(start).Seconds())
			written.Add(1)
			m.TradesWritten.Inc()
			// the trade's own timestamp, so a stale feed shows up as growing lag
			m.LastTradeTime.Set(float64(t.Time.UnixMilli()) / 1000)
		}
	}()

	go func() {
		tick := time.NewTicker(30 * time.Second)
		defer tick.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				log.Printf("stats: %d trades written, %d failed", written.Load(), failed.Load())
			}
		}
	}()

	list := strings.Split(*products, ",")
	log.Printf("gateway starting: products=%v redis=%s stream=%s", list, *redisURL, *stream)
	err = feed.Stream(ctx, feed.DefaultURL, list, trades, feed.WithOnReconnect(m.Reconnects.Inc))
	close(trades)
	if err != nil && ctx.Err() == nil {
		log.Fatalf("feed stopped: %v", err)
	}
	log.Printf("gateway stopped, %d trades written", written.Load())
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
