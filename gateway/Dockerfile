# Go gateway: Coinbase websocket -> Redis Streams.
#   docker build -t quantlab-gateway gateway

FROM golang:1.26-alpine AS build
WORKDIR /src

# download modules first so this layer is cached until go.mod / go.sum change
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# static binary (no cgo) so it can run on a bare alpine; -s -w strips debug info
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.21
# CA certs are needed for the wss:// connection to Coinbase
RUN apk add --no-cache ca-certificates \
    && adduser -D -u 10001 quantlab
USER quantlab
COPY --from=build /out/gateway /usr/local/bin/gateway

# Prometheus /metrics
EXPOSE 9100
# redis url, stream name and metrics address come from QUANTLAB_* env vars (see main.go)
ENTRYPOINT ["gateway"]
