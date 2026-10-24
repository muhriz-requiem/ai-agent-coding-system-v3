"""Tests for the Alpaca paper broker client (src/broker/alpaca.py).

Transport is fully mocked via ``AlpacaClient._post`` / ``_get`` seams.
Rate limiting uses an injectable clock for determinism.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.broker.alpaca", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import pytest

from src.agents.models import OrderPlan
from src.broker.alpaca import AlpacaClient, RateLimiter


def _order(ticker="AAPL", side="buy", qty=10, price=227.5) -> OrderPlan:
    return OrderPlan.model_validate({
        "ticker": ticker, "side": side, "quantity": qty,
        "limit_price": price, "rationale": "test",
    })


def test_paper_endpoint_hard_lock():
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    assert "paper" in client.base_url
    with pytest.raises(ValueError):
        AlpacaClient(key="k", secret="s", paper_only=False)


def test_buy_orders_submit_dollar_notional(monkeypatch):
    """BUYs submit notional (dollar) payloads: exact budget spend, no
    qty/price drift between sizing and execution."""
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    posted = []

    def fake_post(path, payload):
        posted.append((path, payload))
        return {"id": "order-1", "status": "accepted"}

    monkeypatch.setattr(client, "_post", fake_post)
    responses = client.submit_limit_orders(
        [_order(side="buy", qty=12.3456789, price=227.5)])
    assert len(responses) == 1
    path, payload = posted[0]
    assert "/orders" in path
    assert payload["type"] == "market"
    assert payload["side"] == "buy"
    assert payload["symbol"] == "AAPL"
    assert "qty" not in payload
    # 12.3456789 * 227.5 = 2808.64194975 -> ceil to the cent.
    assert payload["notional"] == "2808.65"
    assert "limit_price" not in payload


def test_buy_notional_ceils_to_cent(monkeypatch):
    """Ceil (never floor) so the payload can't land below Alpaca's minimum."""
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    posted = []
    monkeypatch.setattr(client, "_post",
                        lambda path, payload: posted.append(payload) or {})
    # 1.5 * 10.0 = 15.0 exactly -> "15.00" (ceil is a no-op on exact cents).
    client.submit_limit_orders([_order(side="buy", qty=1.5, price=10.0)])
    assert posted[0]["notional"] == "15.00"
    # 0.000001 * 100.0 = 0.0001 -> ceil to "0.01".
    client.submit_limit_orders([_order(side="buy", qty=0.000001, price=100.0)])
    assert posted[1]["notional"] == "0.01"


def test_sell_orders_stay_quantity_based(monkeypatch):
    """SELLs keep fractional qty: exits must flatten the exact position."""
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    posted = []

    def fake_post(path, payload):
        posted.append((path, payload))
        return {"id": "order-2", "status": "accepted"}

    monkeypatch.setattr(client, "_post", fake_post)
    client.submit_limit_orders(
        [_order(side="sell", qty=12.3456789, price=227.5)])
    path, payload = posted[0]
    assert "/orders" in path
    assert payload["type"] == "market"
    assert payload["side"] == "sell"
    assert "notional" not in payload
    assert float(payload["qty"]) == 12.345679


def test_get_account_uses_get_seam(monkeypatch):
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    monkeypatch.setattr(client, "_get", lambda path: {"equity": "12345.67"})
    acct = client.get_account()
    assert acct["equity"] == "12345.67"


def test_rejects_non_orderplan_inputs():
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    with pytest.raises(ValueError):
        client.submit_limit_orders([{"ticker": "AAPL", "side": "buy"}])


def test_get_positions_uses_get_seam(monkeypatch):
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    monkeypatch.setattr(client, "_get",
                        lambda path: [{"symbol": "AAPL", "qty": "10"}])
    assert client.get_positions()[0]["symbol"] == "AAPL"


def test_get_clock_uses_get_seam(monkeypatch):
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    seen = {}

    def fake_get(path):
        seen["path"] = path
        return {"is_open": False, "next_open": "2026-09-08T13:30:00Z",
                "next_close": "2026-09-08T20:00:00Z"}

    monkeypatch.setattr(client, "_get", fake_get)
    clock = client.get_clock()
    assert "/v2/clock" in seen["path"]
    assert clock["is_open"] is False
    assert clock["next_open"].endswith("Z")


def test_get_open_orders_uses_get_seam(monkeypatch):
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    seen = {}

    def fake_get(path):
        seen["path"] = path
        return [{"symbol": "AAPL", "side": "buy", "status": "open"}]

    monkeypatch.setattr(client, "_get", fake_get)
    rows = client.get_open_orders()
    assert "status=open" in seen["path"]
    assert rows[0]["symbol"] == "AAPL"


class FakeClock:
    """Deterministic monotonic clock in 10-second steps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 10.0
        return self.now


def test_rate_limiter_blocks_over_max():
    limiter = RateLimiter(max_per_minute=2, clock=FakeClock())
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == 0.0
    # Third call within the same minute must require a wait.
    wait = limiter.acquire()
    assert wait > 0.0


def test_rate_limiter_window_expires():
    clock = FakeClock()
    limiter = RateLimiter(max_per_minute=2, clock=clock)
    limiter.acquire()
    limiter.acquire()
    clock.now += 120.0  # push past the 60s window
    assert limiter.acquire() == 0.0


def test_rate_limiter_wired_into_client():
    client = AlpacaClient(key="k", secret="s", paper_only=True)
    assert isinstance(client.rate_limiter, RateLimiter)
    assert client.rate_limiter.max_per_minute <= 200
