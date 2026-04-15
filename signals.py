"""
Signal feeds — three data sources feeding the regime classifier.

Signal 1: BTC spot polling (price, momentum, velocity, funding)
Signal 2: Polymarket public APIs (rotating market discovery + orderbook)
Signal 3: Kalshi public API (cross-platform price comparison)
"""

import asyncio
import json
import logging
import ssl
import time
from collections import deque
from datetime import datetime

import aiohttp
import certifi

import config

log = logging.getLogger("signals")


def build_connector():
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return aiohttp.TCPConnector(ssl=ssl_ctx)


def parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def seconds_to_expiry(end_date: str) -> float | None:
    if not end_date:
        return None
    try:
        end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
        return end_ts - time.time()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  SIGNAL 1: BINANCE BTC PRICE FEED
# ═══════════════════════════════════════════════════════

class BinanceFeed:
    """
    BTC/USD real-time price feed via HTTPS polling.

    Uses Coinbase and Kraken public endpoints for price.
    Binance funding remains a best-effort orthogonal signal and may
    be unavailable in some hosting regions.
    """

    PRICE_FEEDS = [
        ("coinbase_exchange", "https://api.exchange.coinbase.com/products/BTC-USD/ticker"),
        ("coinbase_spot", "https://api.coinbase.com/v2/prices/BTC-USD/spot"),
        ("kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"),
    ]

    def __init__(self):
        self.price = None
        self.history = deque(maxlen=500)  # (timestamp, price)
        self.connected = False
        self._session = None
        self._task = None
        self._active_feed = None
        self.funding_rate = 0.0
        self._last_funding_poll = 0.0

    async def start(self):
        """Start HTTPS price polling."""
        self._session = aiohttp.ClientSession(connector=build_connector())
        self._task = asyncio.create_task(self._poll_loop())
        log.info("BTC price feed starting...")

    async def stop(self):
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()

    async def _poll_loop(self):
        """Poll price feeds and refresh funding on a slower cadence."""
        while True:
            try:
                now = time.time()
                price = await self._fetch_price()
                if price:
                    self.price = price
                    self.history.append((now, price))
                    if not self.connected:
                        self.connected = True
                        log.info(f"BTC price feed connected via {self._active_feed}: ${price:,.0f}")

                if now - self._last_funding_poll >= config.BINANCE_FUNDING_POLL_INTERVAL_SEC:
                    self._last_funding_poll = now
                    await self._fetch_funding_rate()

                await asyncio.sleep(config.BTC_PRICE_POLL_INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                log.debug(f"BTC price poll error: {e}")
                await asyncio.sleep(3)

    async def _fetch_price(self) -> float:
        """Try each feed endpoint in order until one works."""

        # 1. Coinbase Exchange
        try:
            async with self._session.get(
                self.PRICE_FEEDS[0][1],
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._active_feed = "coinbase_exchange"
                    return float(data["price"])
        except Exception:
            pass

        # 2. Coinbase Spot
        try:
            async with self._session.get(
                self.PRICE_FEEDS[1][1],
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._active_feed = "coinbase_spot"
                    return float(data["data"]["amount"])
        except Exception:
            pass

        # 3. Kraken
        try:
            async with self._session.get(
                self.PRICE_FEEDS[2][1],
                timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._active_feed = "kraken"
                    return float(data["result"]["XXBTZUSD"]["c"][0])
        except Exception:
            pass

        return None

    async def _fetch_funding_rate(self):
        """Best-effort Binance funding rate fetch."""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {"symbol": "BTCUSDT", "limit": 1}
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    log.debug(f"Funding rate API returned {resp.status}")
                    return

                data = await resp.json()
                if data:
                    self.funding_rate = float(data[0].get("fundingRate", 0) or 0)
        except Exception as e:
            log.debug(f"Funding rate fetch failed: {e}")

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
            "btc_funding_rate": self.funding_rate,
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
        self.selected_market_id = None
        self.known_markets = {}

    async def start(self):
        self._session = aiohttp.ClientSession(connector=build_connector())
        await self._refresh_markets()
        log.info(f"Polymarket feed started. {len(self.active_markets)} markets found.")

    async def stop(self):
        if self._session:
            await self._session.close()

    async def _refresh_markets(self):
        """Find the current live 5-minute and 15-minute BTC up/down markets."""
        candidate_markets = await self._collect_candidate_markets()

        deduped = {}
        for market in candidate_markets:
            market_id = market.get("id")
            if market_id:
                deduped[market_id] = market

        tradable_markets = await self._validate_markets(list(deduped.values()))
        self.active_markets = tradable_markets
        self.selected_market_id = tradable_markets[0]["id"] if tradable_markets else None
        self.orderbooks = {
            market["id"]: market.pop("_orderbook")
            for market in tradable_markets
            if "_orderbook" in market
        }
        self._last_market_refresh = time.time()

        for market in tradable_markets:
            self.known_markets[market["id"]] = dict(market)

        if tradable_markets:
            top = tradable_markets[0]
            log.info(
                "Top market: "
                f"{top['question']} (spread: {top['current_spread']:.02f}, "
                f"vol: ${top['volume_24h']:,.0f}, "
                f"secs: {int(top['seconds_remaining'])})"
            )
        else:
            log.warning("No tradable BTC up/down market found")

    async def _collect_candidate_markets(self):
        candidates = []
        candidates.extend(await self._load_target_slug_events())
        candidates.extend(await self._search_event_candidates())
        candidates.extend(await self._search_market_candidates())
        candidates.extend(await self._search_clob_candidates())
        return candidates

    async def _load_target_slug_events(self):
        now = int(time.time())
        epoch_15m = (now // 900) * 900
        epoch_5m = (now // 300) * 300
        target_slugs = [
            f"btc-updown-15m-{epoch_15m}",
            f"btc-updown-15m-{epoch_15m - 900}",
            f"btc-updown-5m-{epoch_5m}",
            f"btc-updown-5m-{epoch_5m - 300}",
        ]

        candidates = []
        for slug in target_slugs:
            try:
                url = f"{config.GAMMA_API_URL}/events"
                async with self._session.get(
                    url,
                    params={"slug": slug},
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    events = await resp.json()
                    for event in events:
                        candidates.extend(self._markets_from_event(event))
            except Exception as e:
                log.debug(f"Exact event lookup failed for {slug}: {e}")
        return candidates

    async def _search_event_candidates(self):
        candidates = []
        for keyword in ["btc-updown-5m", "btc-updown-15m"]:
            try:
                url = f"{config.GAMMA_API_URL}/events"
                params = {"slug_contains": keyword, "closed": "false", "limit": 5}
                async with self._session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    events = await resp.json()
                    for event in events:
                        slug = (event.get("slug") or "").lower()
                        if keyword not in slug:
                            continue
                        candidates.extend(self._markets_from_event(event))
            except Exception as e:
                log.debug(f"Event search failed for {keyword}: {e}")
        return candidates

    async def _search_market_candidates(self):
        candidates = []
        try:
            url = f"{config.GAMMA_API_URL}/markets"
            params = {"active": "true", "tag": "btc", "limit": 100}
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return candidates
                markets = await resp.json()
        except Exception as e:
            log.debug(f"Gamma market search failed: {e}")
            return candidates

        for market in markets:
            parsed = self._parse_gamma_market(market)
            secs = seconds_to_expiry(parsed.get("end_date", ""))
            if secs is None or secs <= 0 or secs > config.POLYMARKET_EVENT_LOOKAHEAD_SEC:
                continue
            candidates.append(parsed)

        return candidates

    async def _search_clob_candidates(self):
        candidates = []
        try:
            url = f"{config.CLOB_API_URL}/markets"
            params = {"limit": config.POLYMARKET_CLOB_MARKET_LIMIT}
            async with self._session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return candidates
                payload = await resp.json()
        except Exception as e:
            log.debug(f"CLOB market search failed: {e}")
            return candidates

        for market in payload.get("data", []):
            question = (market.get("question") or "").lower()
            if "bitcoin up or down" not in question:
                continue
            if not market.get("accepting_orders"):
                continue

            candidates.append({
                "id": market.get("condition_id", ""),
                "gamma_market_id": "",
                "question": market.get("question", ""),
                "slug": market.get("slug", ""),
                "end_date": market.get("end_date_iso", ""),
                "volume_24h": float(market.get("volume_24hr", 0) or 0),
                "yes_token_id": "",
                "no_token_id": "",
                "outcomes": [],
                "clob_token_ids": [],
                "accepting_orders": market.get("accepting_orders", False),
            })

        return candidates

    def _markets_from_event(self, event: dict):
        markets = []
        for market in event.get("markets", []):
            parsed = self._parse_gamma_market(market, event_slug=event.get("slug", ""))
            secs = seconds_to_expiry(parsed.get("end_date", ""))
            if secs is None or secs <= 0 or secs > config.POLYMARKET_EVENT_LOOKAHEAD_SEC:
                continue
            markets.append(parsed)
        return markets

    def _parse_gamma_market(self, market: dict, event_slug: str = "") -> dict:
        token_ids = parse_json_field(market.get("clobTokenIds", []))
        outcomes = parse_json_field(market.get("outcomes", []))
        return {
            "id": market.get("conditionId", ""),
            "gamma_market_id": str(market.get("id", "")),
            "question": market.get("question", ""),
            "slug": market.get("slug", "") or event_slug,
            "end_date": market.get("endDate", ""),
            "volume_24h": float(market.get("volume24hr", 0) or 0),
            "yes_token_id": token_ids[0] if isinstance(token_ids, list) and len(token_ids) >= 1 else "",
            "no_token_id": token_ids[1] if isinstance(token_ids, list) and len(token_ids) >= 2 else "",
            "outcomes": outcomes if isinstance(outcomes, list) else [],
            "clob_token_ids": token_ids if isinstance(token_ids, list) else [],
            "accepting_orders": market.get("acceptingOrders", True),
        }

    async def _validate_markets(self, markets: list[dict]) -> list[dict]:
        tradable = []

        for market in markets:
            secs = seconds_to_expiry(market.get("end_date", ""))
            if secs is None or secs <= 0 or secs > config.POLYMARKET_EVENT_LOOKAHEAD_SEC:
                continue
            if not market.get("accepting_orders", True):
                continue

            book = await self._fetch_orderbook_snapshot(market)
            if not book:
                continue

            signal = self._book_signal(book)
            spread = signal.get("poly_spread")
            if spread is None or spread > config.MIN_TRADABLE_SPREAD:
                continue

            enriched = {**market, **signal}
            enriched["seconds_remaining"] = secs
            enriched["current_spread"] = spread
            enriched["_orderbook"] = book
            tradable.append(enriched)

        tradable.sort(
            key=lambda market: (
                market.get("current_spread", 999),
                -market.get("volume_24h", 0),
                market.get("seconds_remaining", 999999),
            )
        )
        return tradable

    async def _fetch_orderbook_snapshot(self, market: dict):
        yes_id = market.get("yes_token_id", "")
        if not yes_id:
            return None

        try:
            url = f"{config.CLOB_API_URL}/book"
            async with self._session.get(
                url,
                params={"token_id": yes_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                book = await resp.json()
        except Exception as e:
            log.debug(f"Orderbook fetch failed for {market.get('slug')}: {e}")
            return None

        return {
            "timestamp": time.time(),
            "bids": book.get("bids", []),
            "asks": book.get("asks", []),
            "market": market,
        }

    def _book_signal(self, book_data: dict) -> dict:
        bids = sorted(
            book_data.get("bids", []),
            key=lambda level: float(level.get("price", 0) or 0),
            reverse=True,
        )
        asks = sorted(
            book_data.get("asks", []),
            key=lambda level: float(level.get("price", 0) or 0),
        )
        market = book_data.get("market", {})

        signal = {
            "poly_market_id": market.get("id", ""),
            "poly_yes_best_bid": None,
            "poly_yes_best_ask": None,
            "poly_no_best_bid": None,
            "poly_no_best_ask": None,
            "poly_mid_price": None,
            "poly_spread": None,
            "poly_orderbook_imbalance": None,
            "poly_volume_24h": market.get("volume_24h", 0),
            "poly_seconds_remaining": seconds_to_expiry(market.get("end_date", "")),
        }

        if bids:
            signal["poly_yes_best_bid"] = float(bids[0].get("price", 0))
        if asks:
            signal["poly_yes_best_ask"] = float(asks[0].get("price", 0))

        if signal["poly_yes_best_ask"] is not None:
            signal["poly_no_best_bid"] = round(1.0 - signal["poly_yes_best_ask"], 4)
        if signal["poly_yes_best_bid"] is not None:
            signal["poly_no_best_ask"] = round(1.0 - signal["poly_yes_best_bid"], 4)

        if signal["poly_yes_best_bid"] is not None and signal["poly_yes_best_ask"] is not None:
            signal["poly_mid_price"] = round(
                (signal["poly_yes_best_bid"] + signal["poly_yes_best_ask"]) / 2, 4
            )
            signal["poly_spread"] = round(
                signal["poly_yes_best_ask"] - signal["poly_yes_best_bid"], 4
            )

        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            signal["poly_orderbook_imbalance"] = round(
                (bid_depth - ask_depth) / total_depth, 4
            )

        return signal

    async def update(self):
        """Refresh markets periodically and fetch orderbooks."""
        if time.time() - self._last_market_refresh > config.MARKET_REFRESH_INTERVAL:
            await self._refresh_markets()
        else:
            for market in self.active_markets:
                book = await self._fetch_orderbook_snapshot(market)
                if book:
                    self.orderbooks[market["id"]] = book

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
        preferred_market_id = market_id or self.selected_market_id
        if preferred_market_id and preferred_market_id in self.orderbooks:
            book_data = self.orderbooks[preferred_market_id]
        elif market_id and market_id in self.orderbooks:
            book_data = self.orderbooks[market_id]
        elif self.orderbooks:
            book_data = max(self.orderbooks.values(), key=lambda x: x["timestamp"])
        else:
            return result
        return self._book_signal(book_data)


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

    async def start(self):
        self._session = aiohttp.ClientSession(connector=build_connector())
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
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
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
