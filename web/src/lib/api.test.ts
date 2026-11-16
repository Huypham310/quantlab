import { afterEach, describe, expect, it, vi } from "vitest";
import { API_URL, ApiError, getBars, listSymbols, submitPaperOrder } from "./api";

function mockFetch(response: Partial<Response> | Error) {
  const fn =
    response instanceof Error
      ? vi.fn().mockRejectedValue(response)
      : vi.fn().mockResolvedValue({ ok: true, status: 200, statusText: "OK", ...response });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("lists symbols", async () => {
    const fetch = mockFetch({
      json: async () => [{ symbol: "BTC-USD", timeframe: "1h", bars: 100 }],
    });
    const rows = await listSymbols();
    expect(rows[0].symbol).toBe("BTC-USD");
    expect(fetch.mock.calls[0][0]).toBe(`${API_URL}/symbols`);
  });

  it("builds the bars query string", async () => {
    const fetch = mockFetch({ json: async () => [] });
    await getBars("btc-usd", { timeframe: "1d", limit: 100 });
    expect(fetch.mock.calls[0][0]).toBe(
      `${API_URL}/bars/btc-usd?timeframe=1d&source=coinbase&limit=100`,
    );
  });

  it("posts a paper order as json", async () => {
    const fetch = mockFetch({
      status: 201,
      json: async () => ({ id: 1, status: "filled", symbol: "BTC-USD" }),
    });
    await submitPaperOrder({ symbol: "BTC-USD", side: "buy", qty: 0.01 });
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe(`${API_URL}/paper/orders`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toMatchObject({ side: "buy", qty: 0.01 });
  });

  it("turns network failures into ApiError", async () => {
    mockFetch(new TypeError("Failed to fetch"));
    const err = await listSymbols().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
  });
});
