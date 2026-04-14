"""
FastAPI server — serves dashboard + exposes bot data as JSON endpoints.
Runs alongside the bot in the same process.
"""

import time
import json
import os
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import sqlite3

import config
import database

app = FastAPI(title="Polymarket Bot Dashboard", docs_url="/docs")

base_dir = os.path.dirname(__file__)
static_dir = os.path.join(base_dir, "static")
dashboard_file = os.path.join(base_dir, "index.html")

# Serve static frontend assets if a dedicated static directory exists.
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Shared state (set by main.py) ──
_bot_ref = None

def set_bot_reference(bot):
    global _bot_ref
    _bot_ref = bot


# ── Routes ──

@app.get("/")
async def dashboard():
    return FileResponse(dashboard_file)


@app.get("/api/status")
async def get_status():
    """Live bot status — signals, mode, connections."""
    if not _bot_ref:
        return {"error": "Bot not started"}

    bot = _bot_ref
    risk = bot.risk.get_stats()
    trader = bot.trader.get_stats()
    tick_count = database.get_tick_count(bot.db)

    sig1 = bot.binance.get_signal()
    sig2 = bot.polymarket.get_signal()
    news = bot.news_llm.get_signal()

    # Current regime
    combined = {
        **sig1, **sig2,
        "short_momentum": bot.binance.get_short_momentum(10),
        **news,
        "cross_platform_spread": 0,
    }
    decision = bot.classifier.classify(combined)

    return {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - bot._start_time if hasattr(bot, "_start_time") else 0,
        "tick_count": tick_count,
        "connections": {
            "binance": bot.binance.connected,
            "polymarket": len(bot.polymarket.active_markets),
            "kalshi": len(bot.kalshi.markets),
            "news_llm": bot.news_llm.active,
        },
        "signals": {
            "btc_price": sig1.get("btc_price"),
            "btc_momentum": sig1.get("btc_momentum"),
            "btc_direction": sig1.get("btc_direction"),
            "poly_mid_price": sig2.get("poly_mid_price"),
            "poly_spread": sig2.get("poly_spread"),
            "poly_orderbook_imbalance": sig2.get("poly_orderbook_imbalance"),
            "news_direction": news.get("news_direction"),
            "news_edge": news.get("news_edge"),
            "news_confidence": news.get("news_confidence"),
            "news_call_count": news.get("news_call_count"),
        },
        "regime": {
            "mode": decision.mode,
            "lean_direction": decision.lean_direction,
            "lean_confidence": decision.lean_confidence,
            "source": decision.source,
            "reason": decision.reason,
        },
        "risk": risk,
        "trader": trader,
    }


@app.get("/api/pnl")
async def get_pnl():
    """Daily P&L history."""
    if not _bot_ref:
        return {"error": "Bot not started"}

    rows = _bot_ref.db.execute(
        "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT 30"
    ).fetchall()

    columns = ["date", "starting_balance", "ending_balance", "total_pnl",
               "num_trades", "num_fills", "mode1_pnl", "mode2_pnl",
               "mode3_pnl", "mode4_pnl", "max_drawdown"]

    return {"days": [dict(zip(columns, row)) for row in rows]}


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Recent paper trades."""
    if not _bot_ref:
        return {"error": "Bot not started"}

    rows = _bot_ref.db.execute(
        """SELECT id, timestamp, market_id, side, order_type, price, size,
                  filled, fill_timestamp, fill_price, pnl, mode,
                  lean_direction, lean_confidence
           FROM paper_trades ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()

    columns = ["id", "timestamp", "market_id", "side", "order_type", "price",
               "size", "filled", "fill_timestamp", "fill_price", "pnl",
               "mode", "lean_direction", "lean_confidence"]

    trades = []
    for row in rows:
        t = dict(zip(columns, row))
        t["timestamp_str"] = datetime.fromtimestamp(t["timestamp"]).strftime("%H:%M:%S")
        t["mode_label"] = {1: "Quiet", 2: "Lean", 3: "Event", 4: "Arb"}.get(t["mode"], "?")
        trades.append(t)

    return {"trades": trades}


@app.get("/api/ticks")
async def get_ticks(limit: int = 100):
    """Recent tick data for charts."""
    if not _bot_ref:
        return {"error": "Bot not started"}

    rows = _bot_ref.db.execute(
        """SELECT timestamp, btc_price, btc_momentum, mode,
                  lean_direction, lean_confidence, poly_mid_price, poly_spread
           FROM ticks ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()

    columns = ["timestamp", "btc_price", "btc_momentum", "mode",
               "lean_direction", "lean_confidence", "poly_mid_price", "poly_spread"]

    return {"ticks": [dict(zip(columns, row)) for row in reversed(rows)]}


@app.get("/api/training")
async def get_training():
    """CatBoost training status and history."""
    if not _bot_ref:
        return {"error": "Bot not started"}

    db = _bot_ref.db

    # Total labeled samples
    total_ticks = database.get_tick_count(db)
    labeled = db.execute(
        "SELECT COUNT(*) FROM ticks WHERE optimal_lean IS NOT NULL"
    ).fetchone()[0]
    unlabeled = db.execute(
        "SELECT COUNT(*) FROM ticks WHERE btc_price_after_60s IS NULL"
    ).fetchone()[0]

    # Training history
    runs = db.execute(
        "SELECT timestamp, num_samples, train_accuracy, test_accuracy, "
        "feature_importance, model_deployed "
        "FROM training_log ORDER BY id DESC LIMIT 10"
    ).fetchall()

    training_runs = []
    for r in runs:
        training_runs.append({
            "timestamp": r[0],
            "timestamp_str": datetime.fromtimestamp(r[0]).strftime("%Y-%m-%d %H:%M") if r[0] else "",
            "num_samples": r[1],
            "train_accuracy": r[2],
            "test_accuracy": r[3],
            "feature_importance": json.loads(r[4]) if r[4] else {},
            "model_deployed": bool(r[5]),
        })

    # Current classifier source
    source = _bot_ref.classifier.runtime_source()
    model_exists = os.path.exists(config.MODEL_PATH)

    return {
        "total_ticks": total_ticks,
        "labeled_ticks": labeled,
        "unlabeled_ticks": unlabeled,
        "min_samples_needed": config.MIN_SAMPLES_TO_TRAIN,
        "progress_pct": round(min(100, labeled / max(1, config.MIN_SAMPLES_TO_TRAIN) * 100), 1),
        "classifier_source": source,
        "model_exists": model_exists,
        "model_loaded": _bot_ref.classifier._use_model,
        "model_prediction_ok": _bot_ref.classifier._model_prediction_ok,
        "model_error": _bot_ref.classifier._model_error,
        "training_runs": training_runs,
    }


@app.get("/api/mode_distribution")
async def get_mode_distribution():
    """How often each mode has been active."""
    if not _bot_ref:
        return {"error": "Bot not started"}

    rows = _bot_ref.db.execute(
        """SELECT mode, COUNT(*) as count FROM ticks
           GROUP BY mode ORDER BY mode"""
    ).fetchall()

    total = sum(r[1] for r in rows)
    return {
        "modes": [
            {
                "mode": r[0],
                "label": {1: "Quiet", 2: "Lean", 3: "Event", 4: "Arb"}.get(r[0], "?"),
                "count": r[1],
                "pct": round(r[1] / max(1, total) * 100, 1),
            }
            for r in rows
        ],
        "total": total,
    }
