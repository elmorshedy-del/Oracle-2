"""
Polymarket Paper Trading Bot — Main Orchestrator

Ties together:
- Signal feeds (Binance, Polymarket, Kalshi)
- Regime classifier (deterministic rules → CatBoost)
- Paper trader (simulated execution)
- Risk manager (position limits, drawdown)
- Database logger (SQLite)
- Auto-tuner (CatBoost training pipeline)

Run: python main.py
"""

import asyncio
import json
import time
import os
import sys
import signal
import logging
from statistics import stdev
from collections import deque
from datetime import datetime, timezone

import aiohttp

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import config
import database
from signals import BinanceFeed, PolymarketFeed, KalshiFeed
from news_llm import NewsLLMSignal
from engine import RegimeClassifier, PaperTrader, RiskManager
from tuner import run_tuning_cycle

# ═══════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ═══════════════════════════════════════════
#  MAIN BOT
# ═══════════════════════════════════════════

class PolymarketBot:
    def __init__(self):
        # Database
        self.db = database.init_db()

        # Signals
        self.binance = BinanceFeed()
        self.polymarket = PolymarketFeed()
        self.kalshi = KalshiFeed()
        self.news_llm = NewsLLMSignal()

        # Engine
        self.classifier = RegimeClassifier()
        self.risk = RiskManager(config.STARTING_CAPITAL)
        self.trader = PaperTrader(self.risk)

        # State
        self.running = True
        self.tick_count = 0
        self._start_time = time.time()
        self.price_history = deque(maxlen=50000)  # for backfill
        self.last_tuning_time = 0
        self.tuning_interval = 300  # run tuner every 5 minutes
        self.last_status_time = 0
        self.status_interval = 30  # print status every 30 seconds
        self.last_settlement_check = 0
        self.settlement_interval = config.SETTLEMENT_CHECK_INTERVAL_SEC
        self.last_day = None
        self._last_mid = 0.50  # previous tick's mid price (for fill simulation)
        self._btc_reference_price = None
        self._btc_reference_time = None

        # Load model if exists
        if config.USE_MODEL_IF_AVAILABLE and os.path.exists(config.MODEL_PATH):
            self.classifier.load_model(config.MODEL_PATH)

    async def start(self):
        """Start all feeds and run main loop."""
        log.info("=" * 60)
        log.info("  POLYMARKET PAPER TRADING BOT")
        log.info(f"  Mode: {'PAPER' if config.PAPER_MODE else '⚠️  LIVE ⚠️'}")
        log.info(f"  Capital: ${config.STARTING_CAPITAL:,.0f}")
        log.info("=" * 60)

        # Start signal feeds
        await self.binance.start()
        await self.polymarket.start()
        await self.kalshi.start()
        await self.news_llm.start()

        # Wait for initial data
        log.info("Waiting for initial data...")
        await asyncio.sleep(3)

        # Main loop
        try:
            while self.running:
                await self._tick()
                await asyncio.sleep(config.POLL_INTERVAL_SEC)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _tick(self):
        """
        One cycle of the main loop.

        FIXED lifecycle order:
        1. Simulate fills on EXISTING orders (they've had time to work)
        2. Expire stale orders (age-based, not cancel-all)
        3. Gather signals + classify regime
        4. Replenish quotes only if needed
        5. Check for market settlements
        6. Log everything
        """
        self.tick_count += 1
        now = time.time()

        # ── Daily reset ──
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.last_day:
            if self.last_day is not None:
                self._save_daily_pnl()
            self.risk.reset_daily()
            self.last_day = today

        # ── Step 1: Simulate fills on EXISTING orders ──
        # This runs BEFORE placing new orders so old quotes get
        # a real chance to fill based on price movement since last tick
        pre_sig1 = self.binance.get_signal()
        pre_sig2 = self.polymarket.get_signal()
        fill_market_id = pre_sig2.get("poly_market_id")
        fill_mid = pre_sig2.get("poly_mid_price")
        if fill_mid is not None and fill_market_id:
            self._last_mid = fill_mid
            fills = self.trader.simulate_fills(fill_mid, market_id=fill_market_id)
        else:
            fills = []

        # Log fills to database
        for fill in fills:
            order = fill["order"]
            try:
                database.log_paper_trade(self.db, {
                    "timestamp": order.fill_timestamp,
                    "market_id": order.market_id,
                    "side": order.side,
                    "order_type": order.order_type,
                    "price": order.fill_price,
                    "size": order.size,
                    "mode": order.mode,
                    "lean_direction": order.lean_direction,
                    "lean_confidence": order.lean_confidence,
                })
            except Exception:
                pass

        # ── Step 2: Expire stale orders (NOT cancel-all) ──
        self.trader.expire_stale_orders()

        # ── Step 3: Gather signals + classify ──
        try:
            await self.polymarket.update()
        except Exception as e:
            log.debug(f"Polymarket update error: {e}")

        try:
            await self.kalshi.update()
        except Exception as e:
            log.debug(f"Kalshi update error: {e}")

        sig1 = self.binance.get_signal()
        short_momentum = self.binance.get_short_momentum(seconds=10)

        if sig1["btc_price"]:
            self.price_history.append((now, sig1["btc_price"]))

        sig2 = self.polymarket.get_signal()  # refresh after update
        resolved_mid, mid_source = self._resolve_mid(sig1, sig2, now)
        if resolved_mid is not None:
            sig2["poly_mid_price"] = resolved_mid
            self._last_mid = resolved_mid
        sig2["mid_source"] = mid_source

        sig3 = self.kalshi.get_signal(poly_mid=sig2.get("poly_mid_price"))

        try:
            await self.news_llm.update(poly_mid=sig2.get("poly_mid_price"))
        except Exception as e:
            log.debug(f"News/LLM update error: {e}")
        sig4 = self.news_llm.get_signal()

        combined = {
            **sig1, **sig2, **sig3, **sig4,
            "short_momentum": short_momentum,
        }
        combined.update(self._compute_feature_block(sig1))

        decision = self.classifier.classify(combined)

        # ── Step 4: Replenish quotes only if needed ──
        orders = []
        self._cleanup_synthetic_position()
        if not self.risk.halted:
            orders = self.trader.generate_orders(decision, sig2)

        # Log new orders
        for order in orders:
            try:
                order.db_trade_id = database.log_paper_trade(self.db, {
                    "timestamp": order.timestamp,
                    "market_id": order.market_id,
                    "side": order.side,
                    "order_type": order.order_type,
                    "price": order.price,
                    "size": order.size,
                    "mode": order.mode,
                    "lean_direction": order.lean_direction,
                    "lean_confidence": order.lean_confidence,
                })
            except Exception:
                pass

        # ── Step 5: Settle resolved markets ──
        if now - self.last_settlement_check > self.settlement_interval:
            self.last_settlement_check = now
            await self._settle_resolved_markets()

        # ── Step 6: Log tick ──
        tick_data = {
            **combined,
            "timestamp": now,
            "mode": decision.mode,
            "lean_direction": decision.lean_direction,
            "lean_confidence": decision.lean_confidence,
            "classifier_source": decision.source,
        }
        try:
            tick_id = database.log_tick(self.db, tick_data)
        except Exception as e:
            log.error(f"DB log error: {e}")

        # ── Run tuner periodically ──
        if config.AUTO_TRAINING_ENABLED and now - self.last_tuning_time > self.tuning_interval:
            self.last_tuning_time = now
            try:
                tuner_status = run_tuning_cycle(
                    self.db, list(self.price_history)
                )
                if tuner_status.get("trained"):
                    result = tuner_status["training_result"]
                    if result.get("deployed") and config.USE_MODEL_IF_AVAILABLE:
                        self.classifier.load_model(config.MODEL_PATH)
                        log.info("♟  Switched to CatBoost model")
            except Exception as e:
                log.error(f"Tuner error: {e}")

        # ── Print status ──
        if now - self.last_status_time > self.status_interval:
            self.last_status_time = now
            self._print_status(decision, sig1, fills)

    async def _fetch_confirmed_winner(self, market: dict):
        """Use the Gamma market API to confirm the resolved winner."""
        if not self.polymarket._session:
            return None

        lookup_params = []
        if market.get("id"):
            lookup_params.append({"condition_id": market["id"]})
        if market.get("gamma_market_id"):
            lookup_params.append({"id": market["gamma_market_id"]})
        if market.get("slug"):
            lookup_params.append({"slug": market["slug"]})

        for params in lookup_params:
            try:
                async with self.polymarket._session.get(
                    f"{config.GAMMA_API_URL}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    rows = await resp.json()
            except Exception as e:
                log.debug(f"Settlement lookup failed for {market.get('slug')}: {e}")
                continue

            winner = self._winner_from_gamma_rows(rows)
            if winner:
                return winner
        return None

    def _winner_from_gamma_rows(self, rows):
        if not rows:
            return None

        for resolved in rows:
            explicit_outcome = str(resolved.get("outcome") or "").strip().lower()
            if explicit_outcome in {"yes", "up"}:
                return "YES"
            if explicit_outcome in {"no", "down"}:
                return "NO"

            outcome_prices = resolved.get("outcomePrices", [])
            outcomes = resolved.get("outcomes", [])
            try:
                prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                names = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                if not isinstance(prices, list) or not isinstance(names, list):
                    continue
                if not prices or not names or len(prices) != len(names):
                    continue
                numeric_prices = [float(price) for price in prices]
            except Exception:
                continue

            winner_idx = max(range(len(numeric_prices)), key=lambda idx: numeric_prices[idx])
            if numeric_prices[winner_idx] < config.SETTLEMENT_WINNER_MIN_PRICE:
                continue

            label = str(names[winner_idx]).strip().lower()
            if label in {"yes", "up"}:
                return "YES"
            if label in {"no", "down"}:
                return "NO"
        return None

    def _resolve_mid(self, sig1: dict, sig2: dict, now: float):
        """Use a real Polymarket mid when tradable, otherwise fall back to BTC."""
        poly_mid = sig2.get("poly_mid_price")
        poly_spread = sig2.get("poly_spread")
        if poly_mid is not None and (poly_spread is None or poly_spread <= config.MIN_TRADABLE_SPREAD):
            return poly_mid, "polymarket"

        btc_price = sig1.get("btc_price")
        if btc_price:
            if (
                self._btc_reference_price is None
                or self._btc_reference_time is None
                or now - self._btc_reference_time > config.SYNTHETIC_REFERENCE_RESET_SEC
            ):
                self._btc_reference_price = btc_price
                self._btc_reference_time = now

            btc_change = (btc_price - self._btc_reference_price) / self._btc_reference_price
            synthetic_mid = 0.50 + (btc_change * config.SYNTHETIC_MID_SENSITIVITY)
            synthetic_mid = max(config.SYNTHETIC_MID_MIN, min(config.SYNTHETIC_MID_MAX, synthetic_mid))
            return round(synthetic_mid, 4), "btc_synthetic"

        if poly_mid is not None:
            return poly_mid, "polymarket"
        return self._last_mid, "btc_synthetic"

    def _cleanup_synthetic_position(self):
        """Release synthetic fallback exposure so it cannot freeze the bot."""
        if config.SYNTHETIC_MARKET_ID not in self.risk.positions:
            return

        result = self.risk.release_position_at_cost(config.SYNTHETIC_MARKET_ID)
        if not result:
            return

        self.trader.record_settlement(config.SYNTHETIC_MARKET_ID)
        try:
            database.record_market_settlement(self.db, {
                "market_id": config.SYNTHETIC_MARKET_ID,
                "settled_at": time.time(),
                "winning_side": "FLAT",
                "payout_yes": 0.0,
                "payout_no": 0.0,
                "num_yes_shares": result["num_yes_shares"],
                "num_no_shares": result["num_no_shares"],
                "realized_pnl": 0.0,
                "source": "synthetic-flat-unwind",
            })
        except Exception as e:
            log.debug(f"Failed to record synthetic unwind: {e}")

        log.warning(
            "Released synthetic fallback position at cost to free exposure: "
            f"refund=${result['payout']:.2f}"
        )

    def _compute_feature_block(self, sig1: dict):
        """Build volatility, structure, momentum, and time features."""
        history = list(self.binance.history)
        btc_price = sig1.get("btc_price")

        prices_15 = [price for _, price in history[-15:]]
        prices_60 = [price for _, price in history[-60:]]
        prices_300 = [price for _, price in history[-300:]]

        btc_volatility_15 = self._volatility_from_prices(prices_15, minimum_points=5)
        btc_volatility_60 = self._volatility_from_prices(prices_60, minimum_points=10)
        btc_vol_ratio = (
            btc_volatility_15 / btc_volatility_60
            if btc_volatility_60 > 0
            else 1.0
        )

        if prices_300 and btc_price:
            session_high = max(prices_300)
            session_low = min(prices_300)
            dist_from_high = (session_high - btc_price) / btc_price
            dist_from_low = (btc_price - session_low) / btc_price
        else:
            dist_from_high = 0.0
            dist_from_low = 0.0

        momentum_5s = self.binance.get_short_momentum(5)
        momentum_30s = self.binance.get_short_momentum(30)
        momentum_60s = self.binance.get_short_momentum(60)
        momentum_divergence = int(
            momentum_5s != 0
            and momentum_60s != 0
            and ((momentum_5s > 0) != (momentum_60s > 0))
        )

        now_utc = datetime.now(timezone.utc)
        hour_of_day = now_utc.hour + now_utc.minute / 60.0
        day_of_week = now_utc.weekday()
        is_us_market_hours = int(13.5 <= hour_of_day <= 20.0)

        return {
            "btc_volatility_15": btc_volatility_15,
            "btc_volatility_60": btc_volatility_60,
            "btc_vol_ratio": btc_vol_ratio,
            "dist_from_high": dist_from_high,
            "dist_from_low": dist_from_low,
            "momentum_5s": momentum_5s,
            "momentum_30s": momentum_30s,
            "momentum_divergence": momentum_divergence,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_us_market_hours": is_us_market_hours,
        }

    @staticmethod
    def _volatility_from_prices(prices: list[float], minimum_points: int):
        if len(prices) < minimum_points:
            return 0.0

        returns = []
        for idx in range(1, len(prices)):
            previous = prices[idx - 1]
            current = prices[idx]
            if previous:
                returns.append((current - previous) / previous)

        if len(returns) <= 1:
            return 0.0

        return stdev(returns)
    def _print_status(self, decision, binance_signal, fills):
        """Print a compact status line."""
        risk_stats = self.risk.get_stats()
        trader_stats = self.trader.get_stats()
        tick_count = database.get_tick_count(self.db)

        btc = binance_signal.get("btc_price")
        btc_str = f"${btc:,.0f}" if btc else "..."
        mom = binance_signal.get("btc_momentum", 0)

        mode_labels = {1: "QUIET  ", 2: "LEAN   ", 3: "EVENT  ", 4: "ARB    "}
        mode_str = mode_labels.get(decision.mode, "???    ")

        lean_arrow = ""
        if decision.lean_direction > 0.1:
            lean_arrow = "↑"
        elif decision.lean_direction < -0.1:
            lean_arrow = "↓"
        else:
            lean_arrow = "─"

        ws_status = "●" if self.binance.connected else "○"

        log.info(
            f"{ws_status} BTC {btc_str} ({mom:+.3%}) │ "
            f"Mode {decision.mode}: {mode_str}{lean_arrow} "
            f"conf={decision.lean_confidence:.2f} │ "
            f"Bal: ${risk_stats['balance']:,.0f} "
            f"(PnL: ${risk_stats['total_pnl']:+,.0f}) │ "
            f"Orders: {trader_stats['open_orders']} open, "
            f"{trader_stats['total_fills']} fills │ "
            f"Ticks: {tick_count}"
        )

        if decision.mode > 1:
            log.info(f"  └─ {decision.reason}")

    def _save_daily_pnl(self):
        """Save end-of-day summary."""
        stats = self.risk.get_stats()
        today = self.last_day or datetime.now().strftime("%Y-%m-%d")
        try:
            database.upsert_daily_pnl(self.db, {
                "date": today,
                "starting_balance": self.risk.daily_starting_balance,
                "ending_balance": stats["balance"],
                "total_pnl": stats["daily_pnl"],
                "num_trades": self.trader.total_orders_placed,
                "num_fills": self.trader.total_fills,
                "mode1_pnl": 0,  # TODO: track per-mode PnL
                "mode2_pnl": 0,
                "mode3_pnl": 0,
                "mode4_pnl": 0,
                "max_drawdown": 0,
            })
        except Exception as e:
            log.error(f"Failed to save daily PnL: {e}")

    async def _settle_resolved_markets(self):
        """Resolve matured markets once Gamma confirms the winning side."""
        market_ids = list(self.risk.positions.keys())
        for market_id in market_ids:
            if database.is_market_settled(self.db, market_id):
                continue

            market = self.polymarket.known_markets.get(market_id)
            if not market:
                continue

            end_date = market.get("end_date", "")
            if not end_date:
                continue

            try:
                end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
            except Exception as e:
                log.debug(f"Settlement parse error for {market_id}: {e}")
                continue

            if time.time() <= end_ts:
                continue

            winner = await self._fetch_confirmed_winner(market)
            if not winner:
                continue

            settlement = {
                "winning_side": winner,
                "payout_yes": 1.0 if winner == "YES" else 0.0,
                "payout_no": 1.0 if winner == "NO" else 0.0,
                "source": "gamma-outcome-prices",
            }
            result = self.risk.settle_market(
                market_id,
                settlement["payout_yes"],
                settlement["payout_no"],
            )
            if not result:
                continue

            self.trader.record_settlement(market_id)
            database.record_market_settlement(self.db, {
                "market_id": market_id,
                "settled_at": time.time(),
                "winning_side": settlement["winning_side"],
                "payout_yes": settlement["payout_yes"],
                "payout_no": settlement["payout_no"],
                "num_yes_shares": result["num_yes_shares"],
                "num_no_shares": result["num_no_shares"],
                "realized_pnl": result["realized_pnl"],
                "source": settlement["source"],
            })
            log.info(
                f"  SETTLED: {market_id[:8]} winner={settlement['winning_side']} "
                f"pnl=${result['realized_pnl']:+.2f}"
            )

    async def shutdown(self):
        """Clean shutdown."""
        log.info("Shutting down...")
        self._save_daily_pnl()
        await self.binance.stop()
        await self.polymarket.stop()
        await self.kalshi.stop()
        await self.news_llm.stop()
        self.db.close()
        log.info("Shutdown complete.")


# ═══════════════════════════════════════════
#  ENTRY POINT — runs FastAPI + bot together
# ═══════════════════════════════════════════

async def main():
    import uvicorn
    from api import app, set_bot_reference

    bot = PolymarketBot()
    set_bot_reference(bot)

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()

    def handle_signal():
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass  # Windows

    # Run uvicorn + bot concurrently
    port = int(os.environ.get("PORT", 8080))

    uvicorn_config = uvicorn.Config(
        app, host="0.0.0.0", port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_config)

    log.info(f"Dashboard: http://localhost:{port}")

    # Run both
    await asyncio.gather(
        server.serve(),
        bot.start(),
    )


if __name__ == "__main__":
    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║  Polymarket 4-Mode Paper Trading Bot      ║")
    print("  ║  All paper mode — no real money at risk   ║")
    print("  ║  Dashboard: http://localhost:8080          ║")
    print("  ╚═══════════════════════════════════════════╝")
    print()
    asyncio.run(main())
