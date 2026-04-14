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
import time
import os
import sys
import signal
import logging
from collections import deque
from datetime import datetime

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
        self.last_day = None

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
        """One cycle of the main loop."""
        self.tick_count += 1
        now = time.time()

        # ── Daily reset ──
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.last_day:
            if self.last_day is not None:
                self._save_daily_pnl()
            self.risk.reset_daily()
            self.last_day = today

        # ── Gather signals ──
        try:
            await self.polymarket.update()
        except Exception as e:
            log.debug(f"Polymarket update error: {e}")

        try:
            await self.kalshi.update()
        except Exception as e:
            log.debug(f"Kalshi update error: {e}")

        # Signal 1: Binance
        sig1 = self.binance.get_signal()
        short_momentum = self.binance.get_short_momentum(seconds=10)

        # Record price history for backfill
        if sig1["btc_price"]:
            self.price_history.append((now, sig1["btc_price"]))

        # Signal 2: Polymarket
        sig2 = self.polymarket.get_signal()

        # Signal 3: Kalshi
        sig3 = self.kalshi.get_signal(poly_mid=sig2.get("poly_mid_price"))

        # Signal 4: News + LLM
        try:
            await self.news_llm.update(poly_mid=sig2.get("poly_mid_price"))
        except Exception as e:
            log.debug(f"News/LLM update error: {e}")
        sig4 = self.news_llm.get_signal()

        # ── Combine signals ──
        combined = {
            **sig1, **sig2, **sig3, **sig4,
            "short_momentum": short_momentum,
        }

        # ── Classify regime ──
        decision = self.classifier.classify(combined)

        # ── Simulate fills ──
        mid = sig2.get("poly_mid_price")
        fills = self.trader.simulate_fills(mid)
        for fill in fills:
            order = fill["order"]
            if order.db_trade_id is not None:
                database.mark_fill(
                    self.db,
                    order.db_trade_id,
                    order.fill_price,
                    round(fill["realized_pnl"] + fill["rebate"], 4),
                )

        # ── Generate paper orders ──
        orders = []
        if not self.risk.halted:
            orders = self.trader.generate_orders(decision, sig2)

        # ── Log tick to database ──
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

        # Log newly placed paper trades
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

        # ── Run tuner periodically ──
        if now - self.last_tuning_time > self.tuning_interval:
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
