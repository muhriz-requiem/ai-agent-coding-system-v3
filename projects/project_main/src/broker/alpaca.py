"""Alpaca paper-trading broker client."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable

from src.agents.models import OrderPlan

PAPER_BASE_URL = "https://paper-api.alpaca.markets"

_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Rolling 60-second-window rate limiter with injectable clock."""

    def __init__(self, max_per_minute: int = 200, clock: Callable[[], float] = time.monotonic):
        self.max_per_minute = max_per_minute
        self.clock = clock
        self._timestamps: deque[float] = deque()

    def acquire(self) -> float:
        now = self.clock()
        while self._timestamps and now - self._timestamps[0] >= _WINDOW_SECONDS:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_minute:
            return _WINDOW_SECONDS - (now - self._timestamps[0])
        self._timestamps.append(now)
        return 0.0


class AlpacaClient:
    """Client locked to Alpaca paper trading only."""

    def __init__(self, key: str, secret: str, paper_only: bool = True):
        if not paper_only:
            raise ValueError("live trading is disabled in this system version")
        self.key = key
        self.secret = secret
        self.base_url = PAPER_BASE_URL
        self.rate_limiter = RateLimiter()

    # Transport seams -------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        import httpx

        self.rate_limiter.acquire()
        headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        }
        response = httpx.post(
            f"{self.base_url}{path}", json=payload, headers=headers, timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    def _get(self, path: str) -> list:
        import httpx

        self.rate_limiter.acquire()
        headers = {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        }
        response = httpx.get(
            f"{self.base_url}{path}", headers=headers, timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    # Public API ------------------------------------------------------

    def submit_limit_orders(self, orders: list[OrderPlan]) -> list[dict]:
        for order in orders:
            if not isinstance(order, OrderPlan):
                raise ValueError("orders must be OrderPlan instances")
        responses = []
        for order in orders:
            if order.side == "buy":
                # Notional (dollar-amount) order: Alpaca fills exactly
                # this dollar value at market, so actual spend always
                # equals the planned budget -- a qty order would drift
                # with the price between sizing (last close) and
                # execution and could breach the cumulative equity cap.
                # Ceil to the cent so flooring can never land below the
                # Alpaca minimum notional.
                notional = math.ceil(order.quantity * order.limit_price * 100.0) / 100.0
                payload = {
                    "symbol": order.ticker,
                    "notional": f"{notional:.2f}",
                    "side": order.side,
                    "type": "market",
                    "time_in_force": "day",
                }
            else:
                # SELLs stay quantity-based: exits must flatten the exact
                # held position, which a stale-price notional cannot
                # guarantee (a dollar sell could leave a residual or
                # oversell when the price moved since sizing).
                qty = str(round(order.quantity, 6))
                payload = {
                    "symbol": order.ticker,
                    "qty": qty,
                    "side": order.side,
                    "type": "market",
                    "time_in_force": "day",
                }
            responses.append(self._post("/v2/orders", payload))
        return responses

    def get_account(self) -> dict:
        return self._get("/v2/account")

    def get_positions(self) -> list:
        return self._get("/v2/positions")

    def get_clock(self) -> dict:
        """Market clock: ``is_open``, ``next_open``, ``next_close``."""
        return self._get("/v2/clock")

    def get_open_orders(self) -> list:
        """Open (unfilled) orders — used to avoid duplicate submissions."""
        return self._get("/v2/orders?status=open")
