"""
Signal feeds — three data sources feeding the regime classifier.

Signal 1: BTC price feed (real-time-ish price, momentum, velocity)
Signal 2: Polymarket public API (orderbook state, market discovery)
Signal 3: Kalshi public API (cross-platform price comparison)
"""

import asyncio
import json
import time
import logging
import ssl
from collections import deque

import aiohttp
import certifi

import config

log = logging.getLogger("signals")


def build_ssl_context():
    """Use certifi so aiohttp trusts the same CA bundle as requests."""
    return ssl.create_default_context(cafile=certifi.where())


# ═══════════════════════════════════════════════════════
#  SIGNAL 1: BINANCE BTC PRICE FEED
# ═══════════════════════════════════════════════════════

class BinanceFeed:
    """
    Maintains a rolling BTC price window from public spot endpoints
    to compute momentum and velocity.
    """

    def __init__(self):
        self.price = None
        self.history = deque(maxlen=500)  # (timestamp, price)
        self.connected = False
        self.source = None
        self._session = None
        self._task = None
        self._ssl_context = build_ssl_context()

    async def start(self):
        """Start background polling."""
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._poll_loop())
        log.info("BTC price feed starting...")

    async def stop(self):
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()

    async def _poll_loop(self):
        """Poll reachable spot endpoints with retry/backoff."""
        backoff = 1
        while True:
            try:
                price, source = await self._fetch_price()
                now = time.time()
                self.price = price
                self.history.append((now, price))

                if source != self.source or not self.connected:
                    log.info(f"BTC price feed connected via {source}")

                self.source = source
                self.connected = True
                backoff = 1
                await asyncio.sleep(config.BTC_PRICE_POLL_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                log.warning(f"BTC price feed unavailable: {e}. Retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _fetch_price(self):
        """Try public spot sources in order until one returns a price."""
        errors = []

        for url in config.BTC_PRICE_FEEDS:
            try:
                async with self._session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"Accept": "application/json"},
                    ssl=self._ssl_context,
                ) as resp:
                    if resp.status != 200:
                        errors.append(f"{url} -> {resp.status}")
                        continue

                    data = await resp.json()
                    price = self._extract_price(url, data)
                    if price:
                        return price, self._source_name(url)
                    errors.append(f"{url} -> missing price")
            except Exception as e:
                errors.append(f"{url} -> {e}")

        raise RuntimeError("; ".join(errors))

    @staticmethod
    def _source_name(url: str) -> str:
        if "coinbase.com/v2" in url:
            return "Coinbase Spot"
        if "exchange.coinbase.com" in url:
            return "Coinbase Exchange"
        if "kraken.com" in url:
            return "Kraken"
        return "BTC Feed"

    @staticmethod
    def _extract_price(url: str, data: dict):
        if "exchange.coinbase.com" in url:
            return float(data.get("price", 0) or 0)
        if "coinbase.com/v2" in url:
            return float(data.get("data", {}).get("amount", 0) or 0)
        if "kraken.com" in url:
            result = data.get("result", {})
            if result:
                first_market = next(iter(result.values()))
                return float(first_market.get("c", [0])[0] or 0)
        return None

    def get_signal(self) -> dict:
        """
        Compute Signal 1 from price history.
        Returns: btc_price, btc_momentum, btc_direction, btc_velocity
        """
        now = time.time()
        result = {
            "btc_price": self.price,
            "btc_momentum": 0.0,
            "btc_direction": 0,
            "btc_velocity": 0.0,
        }

        if not self.history or len(self.history) < 5:
            return result

        # Momentum: price change over last N seconds
        window = config.BINANCE_MOMENTUM_WINDOW
        recent = [(t, p) for t, p in self.history if now - t <= window]

        if len(recent) < 2:
            return result

        oldest_price = recent[0][1]
        latest_price = recent[-1][1]
        time_span = recent[-1][0] - recent[0][0]

        if oldest_price == 0 or time_span == 0:
            return result

        pct_change = (latest_price - oldest_price) / oldest_price
        velocity = pct_change / time_span  # %/second

        # Direction: -1, 0, +1 based on threshold
        direction = 0
        if pct_change > config.BINANCE_DIRECTION_THRESHOLD:
            direction = 1
        elif pct_change < -config.BINANCE_DIRECTION_THRESHOLD:
            direction = -1

        result["btc_momentum"] = round(pct_change, 6)
        result["btc_direction"] = direction
        result["btc_velocity"] = round(velocity, 8)

        return result

    def get_short_momentum(self, seconds=3) -> float:
        """Ultra-short momentum for event detection (Mode 3 proxy)."""
        now = time.time()
        recent = [(t, p) for t, p in self.history if now - t <= seconds]
        if len(recent) < 2:
            return 0.0
        return (recent[-1][1] - recent[0][1]) / recent[0][1]


# ═══════════════════════════════════════════════════════
#  SIGNAL 2: POLYMARKET ORDERBOOK
# ═══════════════════════════════════════════════════════

class PolymarketFeed:
    """
    Fetches active BTC markets and orderbook data from Polymarket's
    public APIs (Gamma for market discovery, CLOB for orderbook).
    No authentication needed for read-only.
    """

    def __init__(self):
        self.active_markets = []  # list of market dicts
        self.orderbooks = {}  # market_id → orderbook dict
        self._session = None
        self._last_market_refresh = 0
        self._ssl_context = build_ssl_context()

    async def start(self):
        self._session = aiohttp.ClientSession()
        await self._refresh_markets()
        log.info(f"Polymarket feed started. {len(self.active_markets)} markets found.")

    async def stop(self):
        if self._session:
            await self._session.close()

    async def _refresh_markets(self):
        """Find active BTC prediction markets via Gamma API."""
        try:
            url = f"{config.GAMMA_API_URL}/markets"
            params = {
                "closed": "false",
                "limit": config.POLYMARKET_MARKET_LIMIT,
                "search": config.POLYMARKET_SEARCH_QUERY,
            }
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=self._ssl_context,
            ) as resp:
                if resp.status == 200:
                    all_markets = await resp.json()
                    btc_markets = []
                    for m in all_markets:
                        if m.get("closed") or not m.get("active", True):
                            continue

                        q = (m.get("question", "") + m.get("description", "")).lower()
                        if not any(tag in q for tag in ["bitcoin", "btc"]):
                            continue

                        token_ids = self._parse_json_field(m.get("clobTokenIds", []))
                        outcomes = self._parse_json_field(m.get("outcomes", []))
                        if not isinstance(token_ids, list) or len(token_ids) < 2:
                            continue

                        btc_markets.append({
                            "id": m.get("conditionId", ""),
                            "question": m.get("question", ""),
                            "slug": m.get("slug", ""),
                            "end_date": m.get("endDate", ""),
                            "volume_24h": float(m.get("volume24hr", 0) or 0),
                            "yes_token_id": token_ids[0],
                            "no_token_id": token_ids[1],
                            "outcomes": outcomes if isinstance(outcomes, list) else [],
                            "clob_token_ids": token_ids,
                        })

                    btc_markets.sort(
                        key=lambda m: (
                            0 if any(term in m["question"].lower() for term in ["5 min", "15 min", "1 hour", "today", "this week"]) else 1,
                            -m["volume_24h"],
                            m["end_date"] or "9999",
                        )
                    )
                    self.active_markets = btc_markets
                    self._last_market_refresh = time.time()
                else:
                    log.warning(f"Gamma API returned {resp.status}")
        except Exception as e:
            log.error(f"Market refresh failed: {e}")

    @staticmethod
    def _parse_json_field(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    async def update(self):
        """Refresh markets periodically and fetch orderbooks."""
        if time.time() - self._last_market_refresh > config.MARKET_REFRESH_INTERVAL:
            await self._refresh_markets()

        for market in self.active_markets:
            await self._fetch_orderbook(market)

    async def _fetch_orderbook(self, market: dict):
        """Fetch orderbook for a specific market from CLOB."""
        yes_id = market.get("yes_token_id", "")
        if not yes_id:
            return
        try:
            url = f"{config.CLOB_API_URL}/book"
            params = {"token_id": yes_id}
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
                ssl=self._ssl_context,
            ) as resp:
                if resp.status == 200:
                    book = await resp.json()
                    self.orderbooks[market["id"]] = {
                        "timestamp": time.time(),
                        "bids": book.get("bids", []),
                        "asks": book.get("asks", []),
                        "market": market,
                    }
        except Exception as e:
            log.debug(f"Orderbook fetch failed for {market['id'][:8]}: {e}")

    def get_signal(self, market_id=None) -> dict:
        """
        Compute orderbook signal for a given market (or best available).
        Returns: best bid/ask for YES/NO, spread, imbalance, etc.
        """
        result = {
            "poly_market_id": None,
            "poly_yes_best_bid": None,
            "poly_yes_best_ask": None,
            "poly_no_best_bid": None,
            "poly_no_best_ask": None,
            "poly_mid_price": None,
            "poly_spread": None,
            "poly_orderbook_imbalance": None,
            "poly_volume_24h": None,
            "poly_seconds_remaining": None,
        }

        # Pick market
        if market_id and market_id in self.orderbooks:
            book_data = self.orderbooks[market_id]
        elif self.orderbooks:
            # Use most recent orderbook
            book_data = max(self.orderbooks.values(), key=lambda x: x["timestamp"])
        else:
            return result

        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])
        market = book_data.get("market", {})

        result["poly_market_id"] = market.get("id", "")

        # YES side
        if bids:
            result["poly_yes_best_bid"] = float(bids[0].get("price", 0))
        if asks:
            result["poly_yes_best_ask"] = float(asks[0].get("price", 0))

        # NO side = 1 - YES (binary market)
        if result["poly_yes_best_ask"] is not None:
            result["poly_no_best_bid"] = round(1.0 - result["poly_yes_best_ask"], 4)
        if result["poly_yes_best_bid"] is not None:
            result["poly_no_best_ask"] = round(1.0 - result["poly_yes_best_bid"], 4)

        # Mid price and spread
        if result["poly_yes_best_bid"] and result["poly_yes_best_ask"]:
            result["poly_mid_price"] = round(
                (result["poly_yes_best_bid"] + result["poly_yes_best_ask"]) / 2, 4
            )
            result["poly_spread"] = round(
                result["poly_yes_best_ask"] - result["poly_yes_best_bid"], 4
            )

        # Orderbook imbalance: positive = more bid pressure (bullish)
        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        total = bid_depth + ask_depth
        if total > 0:
            result["poly_orderbook_imbalance"] = round(
                (bid_depth - ask_depth) / total, 4
            )

        result["poly_volume_24h"] = market.get("volume_24h", 0)

        # Time remaining (rough estimate from end_date)
        end_date = market.get("end_date", "")
        if end_date:
            try:
                from datetime import datetime
                end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
                result["poly_seconds_remaining"] = max(0, end_ts - time.time())
            except Exception:
                pass

        return result


# ═══════════════════════════════════════════════════════
#  SIGNAL 3: KALSHI CROSS-PLATFORM
# ═══════════════════════════════════════════════════════

class KalshiFeed:
    """
    Fetches BTC hourly markets from Kalshi's public API
    for cross-platform price comparison.
    """

    def __init__(self):
        self.markets = []
        self.prices = {}  # event_ticker → {yes_price, no_price}
        self._session = None
        self._last_poll = 0
        self._ssl_context = build_ssl_context()

    async def start(self):
        self._session = aiohttp.ClientSession()
        await self._fetch_markets()
        log.info(f"Kalshi feed started. {len(self.markets)} markets found.")

    async def stop(self):
        if self._session:
            await self._session.close()

    async def _fetch_markets(self):
        """Fetch active BTC markets from Kalshi."""
        try:
            url = f"{config.KALSHI_API_URL}/markets"
            params = {
                "series_ticker": config.KALSHI_SERIES_TICKER,
                "status": "open",
                "limit": 20,
            }
            headers = {"Accept": "application/json"}
            async with self._session.get(url, params=params, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=10),
                                         ssl=self._ssl_context) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.markets = data.get("markets", [])
                else:
                    log.debug(f"Kalshi API returned {resp.status}")
        except Exception as e:
            log.debug(f"Kalshi fetch failed: {e}")

    async def update(self):
        """Poll Kalshi prices."""
        if time.time() - self._last_poll < config.KALSHI_POLL_INTERVAL:
            return
        self._last_poll = time.time()
        await self._fetch_markets()

        for market in self.markets:
            ticker = market.get("ticker", "")
            yes_price = market.get("yes_ask", market.get("last_price", 0))
            no_price = market.get("no_ask", 0)

            # Kalshi prices are in cents, normalize to dollars
            if isinstance(yes_price, (int, float)) and yes_price > 1:
                yes_price = yes_price / 100.0
            if isinstance(no_price, (int, float)) and no_price > 1:
                no_price = no_price / 100.0

            self.prices[ticker] = {
                "yes_price": yes_price,
                "no_price": no_price,
                "strike": market.get("strike_price", ""),
                "close_time": market.get("close_time", ""),
            }

    def get_signal(self, poly_mid=None) -> dict:
        """
        Cross-platform signal: compare best Kalshi price vs Polymarket.
        """
        result = {
            "kalshi_yes_price": None,
            "kalshi_no_price": None,
            "cross_platform_spread": None,
        }

        if not self.prices:
            return result

        # Find the most relevant Kalshi market (closest to expiry)
        best = None
        for ticker, data in self.prices.items():
            if best is None or (data.get("close_time", "") < best.get("close_time", "z")):
                best = data
                best["ticker"] = ticker

        if best:
            result["kalshi_yes_price"] = best.get("yes_price")
            result["kalshi_no_price"] = best.get("no_price")

            # Cross-platform spread: if we can buy YES on one and NO on other
            # for less than $1 combined, that's an arb
            if poly_mid is not None and result["kalshi_yes_price"]:
                # Strategy: buy Poly NO (1 - poly_mid) + Kalshi YES
                poly_no_cost = 1.0 - (poly_mid or 0.5)
                combined = poly_no_cost + (result["kalshi_yes_price"] or 0.5)
                result["cross_platform_spread"] = round(1.0 - combined, 4)
                # Positive spread = arb opportunity

        return result
