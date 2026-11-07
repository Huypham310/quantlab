import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.paper import PaperBroker, RiskLimits

T = 1_700_000_000_000


def broker(**limits) -> PaperBroker:
    b = PaperBroker(cash=100_000.0, limits=RiskLimits(**limits), fee_bps=10, slippage_bps=5)
    b.on_trade("BTC-USD", 50_000.0, T)
    b.on_trade("ETH-USD", 2_000.0, T)
    return b


def test_market_order_fills_on_the_next_trade_with_slippage_and_fees():
    b = broker()
    o = b.submit("BTC-USD", "buy", 0.09, now_ms=T)  # 0.09 x 50.5k < the 5k order limit
    assert o.status == "open" and not b.fills  # nothing fills until the next print
    (fill,) = b.on_trade("BTC-USD", 50_100.0, T + 1)
    assert fill.price == pytest.approx(50_100 * 1.0005)
    assert fill.fee == pytest.approx(0.09 * fill.price * 0.001)
    assert b.positions["BTC-USD"] == pytest.approx(0.09)
    assert b.cash == pytest.approx(100_000 - 0.09 * fill.price - fill.fee)
    assert o.status == "filled" and not b.open


def test_limit_orders_fill_at_their_price_only_when_crossed():
    b = broker()
    buy = b.submit("ETH-USD", "buy", 1.0, "limit", 1_990.0, now_ms=T)
    assert b.on_trade("ETH-USD", 1_995.0, T + 1) == []
    (fill,) = b.on_trade("ETH-USD", 1_985.0, T + 2)
    assert fill.price == 1_990.0  # passive fill at the limit, not the print
    assert buy.status == "filled"
    sell = b.submit("ETH-USD", "sell", 1.0, "limit", 2_010.0, now_ms=T + 3)
    b.on_trade("ETH-USD", 2_015.0, T + 4)
    assert sell.status == "filled" and b.positions["ETH-USD"] == pytest.approx(0.0)


def test_marketable_limit_takes_the_market_price_not_its_limit():
    # a buy limit 1% above the market used to fill *at the limit*, i.e. pay 1% extra every time
    b = broker()
    o = b.submit("ETH-USD", "buy", 1.0, "limit", 2_020.0, now_ms=T)
    assert o.marketable
    (fill,) = b.on_trade("ETH-USD", 2_001.0, T + 1)
    assert fill.price == pytest.approx(2_001.0 * 1.0005)  # print + slippage, below the limit
    # and if the market runs past the limit first, it doesn't trade
    o2 = b.submit("ETH-USD", "sell", 1.0, "limit", 1_990.0, now_ms=T + 2)
    assert o2.marketable and b.on_trade("ETH-USD", 1_980.0, T + 3) == []


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"symbol": "DOGE-USD", "side": "buy", "qty": 1}, "not allowed"),
        ({"symbol": "BTC-USD", "side": "buy", "qty": 0}, "positive"),
        ({"symbol": "BTC-USD", "side": "buy", "qty": 0.2}, "order notional"),  # ~10k > 5k
        (
            {
                "symbol": "BTC-USD",
                "side": "buy",
                "qty": 0.01,
                "type": "limit",
                "limit_price": 40_000,
            },
            "band",
        ),  # fmt: skip
    ],
)
def test_rejections(kwargs, reason):
    o = broker().submit(now_ms=T, **kwargs)
    assert o.status == "rejected" and reason in o.reason


def test_no_price_yet():
    b = PaperBroker()
    assert "no price" in b.submit("BTC-USD", "buy", 0.01, now_ms=T).reason


def test_position_limit_counts_open_orders():
    b = broker(max_order_notional=15_000, max_position_notional=20_000)
    # two resting buys of 0.2 BTC (~10k each) fit, a third would make the worst case 30k
    for _ in range(2):
        assert b.submit("BTC-USD", "buy", 0.2, "limit", 49_900, now_ms=T).status == "open"
    third = b.submit("BTC-USD", "buy", 0.2, "limit", 49_900, now_ms=T)
    assert third.status == "rejected" and third.reason == "position limit"
    # a sell reduces the worst case on the other side and is fine
    assert b.submit("BTC-USD", "sell", 0.05, now_ms=T).status == "open"


def test_gross_leverage_across_symbols():
    b = broker(max_order_notional=20_000, max_position_notional=20_000, max_gross_leverage=0.3)
    assert b.submit("BTC-USD", "buy", 0.3, now_ms=T).status == "open"  # ~15k of 30k allowed
    o = b.submit("ETH-USD", "buy", 9.0, now_ms=T)  # +18k would make 33k > 0.3 x 100k
    assert o.status == "rejected" and "gross" in o.reason


def test_rate_limit_is_a_sliding_minute():
    b = broker(max_orders_per_minute=3)
    for i in range(3):
        assert b.submit("ETH-USD", "buy", 0.1, now_ms=T + i).status == "open"
    assert b.submit("ETH-USD", "buy", 0.1, now_ms=T + 10).reason == "order rate limit"
    assert b.submit("ETH-USD", "buy", 0.1, now_ms=T + 60_001).status == "open"


def test_kill_switch_cancels_and_only_allows_reducing_orders():
    b = broker(max_order_notional=20_000, max_daily_loss=0.05)
    b.submit("BTC-USD", "buy", 0.35, now_ms=T)
    b.on_trade("BTC-USD", 50_000.0, T + 1)  # ~17.5k long
    resting = b.submit("ETH-USD", "buy", 1.0, "limit", 1_950.0, now_ms=T + 2)
    b.on_trade("BTC-USD", 30_000.0, T + 3)  # -40% on BTC: ~ -7k equity, past -5%
    assert b.killed
    assert resting.status == "cancelled" and "kill switch" in resting.reason
    assert "daily loss" in b.submit("ETH-USD", "buy", 0.1, now_ms=T + 4).reason
    assert b.submit("BTC-USD", "sell", 0.35, now_ms=T + 5).status == "open"  # flattening is ok
    b.new_day()
    assert not b.killed


def test_fill_is_clamped_when_the_price_jumps():
    # checked at last x 1.01, but the next print is 8% higher: filling all of it would breach
    b = broker(max_order_notional=20_000, max_position_notional=20_000)
    o = b.submit("BTC-USD", "buy", 0.39, now_ms=T)  # 0.39 x 50.5k = 19.7k, just under
    assert o.status == "open"
    b.on_trade("BTC-USD", 54_000.0, T + 1)
    assert b.positions["BTC-USD"] * b.fills[-1].price <= 20_000 * (1 + 1e-9)
    assert o.status == "cancelled" and 0 < o.filled_qty < 0.39
    assert b.breaches == 0


def test_replay_smoke(capsys):
    from quantlab.paper import replay

    r = replay.run(20_000, seed=3, order_prob=0.1)
    assert r["breaches"] == 0
    assert r["orders"] > 1_000 and r["fills"] > 0
    assert sum(r["outcomes"].values()) == r["orders"]
    replay.main(["--ticks", "2000"])
    assert "breaches: 0" in capsys.readouterr().out


actions = st.lists(
    st.one_of(
        st.tuples(
            st.just("order"),
            st.sampled_from(["BTC-USD", "ETH-USD"]),
            st.sampled_from(["buy", "sell"]),
            # size as a fraction of the order notional limit: raw coin quantities got almost
            # every order rejected, so positions never got near the limit and the test missed
            # a disabled fill clamp
            st.floats(0.05, 1.0),
            st.one_of(st.none(), st.floats(0.96, 1.04)),  # limit price as a multiple of last
        ),
        st.tuples(
            st.just("tick"),
            st.sampled_from(["BTC-USD", "ETH-USD"]),
            st.floats(0.8, 1.25),  # jumps of up to +/-20-25% per print
        ),
        st.tuples(st.just("cancel"), st.integers(1, 200)),
    ),
    # Hypothesis favours short lists; a breach needs an order *and* a later jump on the same
    # symbol, so without a floor it never built one (a disabled fill clamp went unnoticed)
    min_size=20,
    max_size=120,
)


@settings(max_examples=200, deadline=None)
@given(actions)
def test_random_order_flow_never_breaches_and_books_balance(seq):
    # position limit just above one max-size order: one big order + one price jump reaches it
    b = broker(max_order_notional=8_000, max_position_notional=9_000, max_daily_loss=0.5)
    t = T
    for act in seq:
        t += 1_000
        if act[0] == "order":
            _, sym, side, frac, mult = act
            last = b.last[sym]
            qty = frac * 8_000 / (last * 1.02)
            if mult is None:
                b.submit(sym, side, qty, now_ms=t)
            else:
                b.submit(sym, side, qty, "limit", last * mult, now_ms=t)
        elif act[0] == "tick":
            _, sym, mult = act
            b.on_trade(sym, b.last[sym] * mult, t)
        else:
            b.cancel(act[1])
    assert b.breaches == 0
    # every fill that grew a position respected the limit at its own price (reducing fills
    # can happen above it: the price may have run up since the position was opened)
    pos: dict[str, float] = {}
    cash = 100_000.0
    for f in b.fills:
        signed = f.qty if f.side == "buy" else -f.qty
        before = pos.get(f.symbol, 0.0)
        pos[f.symbol] = before + signed
        if abs(pos[f.symbol]) > abs(before):
            assert abs(pos[f.symbol]) * f.price <= 9_000 * (1 + 1e-9)
        cash -= signed * f.price + f.fee
    # the books balance: cash and positions are exactly what the fills say
    assert b.cash == pytest.approx(cash)
    for sym, q in pos.items():
        assert b.positions[sym] == pytest.approx(q, abs=1e-9)
