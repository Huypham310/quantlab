// Package sink writes trades to a Redis Stream.
package sink

import (
	"context"
	"strconv"

	"github.com/Huypham310/quantlab/gateway/internal/feed"
	"github.com/redis/go-redis/v9"
)

// Writer appends trades to one stream. The stream is capped (approximately) at MaxLen entries
// so Redis memory stays bounded; consumers persist what they need to TimescaleDB.
type Writer struct {
	Client *redis.Client
	Stream string
	MaxLen int64
}

func (w *Writer) Write(ctx context.Context, t feed.Trade) error {
	return w.Client.XAdd(ctx, &redis.XAddArgs{
		Stream: w.Stream,
		MaxLen: w.MaxLen,
		// "~" trimming is much cheaper than exact trimming
		Approx: true,
		Values: Fields(t),
	}).Err()
}

// Fields is the flat string map stored per entry; the Python consumer reads these keys.
func Fields(t feed.Trade) map[string]any {
	return map[string]any{
		"exchange": t.Exchange,
		"symbol":   t.Symbol,
		"trade_id": strconv.FormatInt(t.TradeID, 10),
		"price":    strconv.FormatFloat(t.Price, 'f', -1, 64),
		"size":     strconv.FormatFloat(t.Size, 'f', -1, 64),
		"side":     t.Side,
		// milliseconds since epoch, easy to parse on the python side
		"ts_ms": strconv.FormatInt(t.Time.UnixMilli(), 10),
	}
}
