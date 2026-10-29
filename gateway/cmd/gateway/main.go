// Command gateway streams live trades from the Coinbase public feed into Redis Streams.
//
// For now it only parses its flags; the feed and Redis writer come in the stream phase.
package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
)

func main() {
	products := flag.String("products", "BTC-USD,ETH-USD", "comma-separated Coinbase product ids")
	redisURL := flag.String("redis", envOr("QUANTLAB_REDIS_URL", "redis://localhost:6379/0"), "redis url")
	flag.Parse()

	log.Printf("gateway starting: products=%s redis=%s", strings.Split(*products, ","), *redisURL)

	// chờ Ctrl+C rồi thoát, phần stream sẽ viết sau
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	log.Println("gateway stopped")
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
