"""
SQLite logger — records every tick, signal, decision, and paper trade.
This becomes the training dataset for CatBoost auto-tuning.
"""

import sqlite3
import os
import time
import json

import config


def init_db(path=None):
    """Create database and tables if they don't exist."""
    path = path or config.LOG_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # ── Main tick log: every signal reading + decision ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            -- Signal 1: Binance
            btc_price REAL,
            btc_momentum REAL,
            btc_direction INTEGER,  -- -1, 0, +1
            btc_velocity REAL,      -- rate of change
            -- Polymarket orderbook
            poly_market_id TEXT,
            poly_yes_best_bid REAL,
            poly_yes_best_ask REAL,
            poly_no_best_bid REAL,
            poly_no_best_ask REAL,
            poly_mid_price REAL,
            poly_spread REAL,
            poly_orderbook_imbalance REAL,  -- -1 to +1
            poly_volume_24h REAL,
            poly_seconds_remaining REAL,
            -- Signal 3: Cross-platform
            kalshi_yes_price REAL,
            kalshi_no_price REAL,
            cross_platform_spread REAL,
            -- Regime classifier output
            mode INTEGER,  -- 1, 2, 3, 4
            lean_direction REAL,  -- -1.0 to +1.0
            lean_confidence REAL, -- 0.0 to 1.0
            classifier_source TEXT,  -- 'rules' or 'model'
            -- For training: what actually happened next
            btc_price_after_30s REAL,
            btc_price_after_60s REAL,
            btc_price_after_300s REAL,
            optimal_lean REAL  -- computed retroactively by tuner
        )
    """)

    # ── Paper trade log ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            market_id TEXT,
            side TEXT,           -- 'YES' or 'NO'
            order_type TEXT,     -- 'BID' or 'ASK'
            price REAL,
            size REAL,
            filled INTEGER DEFAULT 0,
            fill_timestamp REAL,
            fill_price REAL,
            pnl REAL DEFAULT 0.0,
            mode INTEGER,
            lean_direction REAL,
            lean_confidence REAL
        )
    """)

    # ── Daily P&L summary ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_pnl (
            date TEXT PRIMARY KEY,
            starting_balance REAL,
            ending_balance REAL,
            total_pnl REAL,
            num_trades INTEGER,
            num_fills INTEGER,
            mode1_pnl REAL DEFAULT 0.0,
            mode2_pnl REAL DEFAULT 0.0,
            mode3_pnl REAL DEFAULT 0.0,
            mode4_pnl REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0
        )
    """)

    # ── Model training log ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            num_samples INTEGER,
            train_accuracy REAL,
            test_accuracy REAL,
            feature_importance TEXT,  -- JSON
            model_deployed INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    return conn


def log_tick(conn, data: dict):
    """Insert one tick row. Pass a dict with column names as keys."""
    columns = [
        "timestamp", "btc_price", "btc_momentum", "btc_direction",
        "btc_velocity", "poly_market_id", "poly_yes_best_bid",
        "poly_yes_best_ask", "poly_no_best_bid", "poly_no_best_ask",
        "poly_mid_price", "poly_spread", "poly_orderbook_imbalance",
        "poly_volume_24h", "poly_seconds_remaining",
        "kalshi_yes_price", "kalshi_no_price", "cross_platform_spread",
        "mode", "lean_direction", "lean_confidence", "classifier_source"
    ]
    values = [data.get(c) for c in columns]
    placeholders = ",".join(["?"] * len(columns))
    col_str = ",".join(columns)
    conn.execute(f"INSERT INTO ticks ({col_str}) VALUES ({placeholders})", values)
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def log_paper_trade(conn, data: dict):
    """Insert a paper trade."""
    columns = [
        "timestamp", "market_id", "side", "order_type", "price",
        "size", "mode", "lean_direction", "lean_confidence"
    ]
    values = [data.get(c) for c in columns]
    placeholders = ",".join(["?"] * len(columns))
    col_str = ",".join(columns)
    conn.execute(f"INSERT INTO paper_trades ({col_str}) VALUES ({placeholders})", values)
    conn.commit()


def mark_fill(conn, trade_id, fill_price, pnl):
    """Mark a paper trade as filled."""
    conn.execute(
        "UPDATE paper_trades SET filled=1, fill_timestamp=?, fill_price=?, pnl=? WHERE id=?",
        (time.time(), fill_price, pnl, trade_id)
    )
    conn.commit()


def backfill_future_prices(conn, tick_id, price_30s, price_60s, price_300s):
    """Called retroactively to fill in what BTC actually did after each tick."""
    conn.execute(
        """UPDATE ticks SET btc_price_after_30s=?, btc_price_after_60s=?,
           btc_price_after_300s=? WHERE id=?""",
        (price_30s, price_60s, price_300s, tick_id)
    )
    conn.commit()


def get_tick_count(conn):
    """Total ticks logged."""
    return conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]


def get_training_data(conn):
    """Pull labeled tick data for CatBoost training."""
    query = """
        SELECT
            btc_price, btc_momentum, btc_direction, btc_velocity,
            poly_yes_best_bid, poly_yes_best_ask, poly_no_best_bid,
            poly_no_best_ask, poly_mid_price, poly_spread,
            poly_orderbook_imbalance, poly_volume_24h,
            poly_seconds_remaining,
            kalshi_yes_price, kalshi_no_price, cross_platform_spread,
            btc_price_after_30s, btc_price_after_60s, btc_price_after_300s,
            optimal_lean
        FROM ticks
        WHERE btc_price_after_60s IS NOT NULL
          AND optimal_lean IS NOT NULL
    """
    rows = conn.execute(query).fetchall()
    columns = [
        "btc_price", "btc_momentum", "btc_direction", "btc_velocity",
        "poly_yes_best_bid", "poly_yes_best_ask", "poly_no_best_bid",
        "poly_no_best_ask", "poly_mid_price", "poly_spread",
        "poly_orderbook_imbalance", "poly_volume_24h",
        "poly_seconds_remaining",
        "kalshi_yes_price", "kalshi_no_price", "cross_platform_spread",
        "btc_price_after_30s", "btc_price_after_60s", "btc_price_after_300s",
        "optimal_lean"
    ]
    return rows, columns


def get_unlabeled_ticks(conn, max_age_seconds=600):
    """Get ticks that need future-price backfill (older than 5 min)."""
    cutoff = time.time() - max_age_seconds
    return conn.execute(
        """SELECT id, timestamp, btc_price FROM ticks
           WHERE btc_price_after_300s IS NULL AND timestamp < ?
           ORDER BY timestamp ASC LIMIT 500""",
        (cutoff,)
    ).fetchall()


def get_last_training_row_count(conn):
    """How many samples were used in the last training run."""
    row = conn.execute(
        "SELECT num_samples FROM training_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else 0


def get_daily_pnl(conn, date_str):
    """Get daily P&L record."""
    return conn.execute(
        "SELECT * FROM daily_pnl WHERE date=?", (date_str,)
    ).fetchone()


def upsert_daily_pnl(conn, data: dict):
    """Insert or update daily P&L."""
    conn.execute("""
        INSERT INTO daily_pnl (date, starting_balance, ending_balance,
            total_pnl, num_trades, num_fills, mode1_pnl, mode2_pnl,
            mode3_pnl, mode4_pnl, max_drawdown)
        VALUES (:date, :starting_balance, :ending_balance,
            :total_pnl, :num_trades, :num_fills, :mode1_pnl, :mode2_pnl,
            :mode3_pnl, :mode4_pnl, :max_drawdown)
        ON CONFLICT(date) DO UPDATE SET
            ending_balance=:ending_balance,
            total_pnl=:total_pnl,
            num_trades=:num_trades,
            num_fills=:num_fills,
            mode1_pnl=:mode1_pnl,
            mode2_pnl=:mode2_pnl,
            mode3_pnl=:mode3_pnl,
            mode4_pnl=:mode4_pnl,
            max_drawdown=:max_drawdown
    """, data)
    conn.commit()
