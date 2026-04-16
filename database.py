"""
SQLite logger — records every tick, signal, decision, and paper trade.
This becomes the training dataset for CatBoost auto-tuning.
"""

import sqlite3
import os
import time
import json

import config
from model_features import model_feature_names


TICK_COLUMNS = [
    ("timestamp", "REAL NOT NULL"),
    ("btc_price", "REAL"),
    ("btc_momentum", "REAL"),
    ("btc_direction", "INTEGER"),
    ("btc_velocity", "REAL"),
    ("poly_market_id", "TEXT"),
    ("poly_yes_best_bid", "REAL"),
    ("poly_yes_best_ask", "REAL"),
    ("poly_no_best_bid", "REAL"),
    ("poly_no_best_ask", "REAL"),
    ("poly_mid_price", "REAL"),
    ("poly_spread", "REAL"),
    ("poly_orderbook_imbalance", "REAL"),
    ("poly_volume_24h", "REAL"),
    ("poly_seconds_remaining", "REAL"),
    ("btc_volatility_15", "REAL DEFAULT 0"),
    ("btc_volatility_60", "REAL DEFAULT 0"),
    ("btc_vol_ratio", "REAL DEFAULT 1.0"),
    ("dist_from_high", "REAL DEFAULT 0"),
    ("dist_from_low", "REAL DEFAULT 0"),
    ("momentum_5s", "REAL DEFAULT 0"),
    ("momentum_30s", "REAL DEFAULT 0"),
    ("momentum_divergence", "INTEGER DEFAULT 0"),
    ("hour_of_day", "REAL DEFAULT 0"),
    ("day_of_week", "INTEGER DEFAULT 0"),
    ("is_us_market_hours", "INTEGER DEFAULT 0"),
    ("btc_funding_rate", "REAL DEFAULT 0"),
    ("mid_source", "TEXT DEFAULT 'polymarket'"),
    ("kalshi_yes_price", "REAL"),
    ("kalshi_no_price", "REAL"),
    ("cross_platform_spread", "REAL"),
    ("mode", "INTEGER"),
    ("lean_direction", "REAL"),
    ("lean_confidence", "REAL"),
    ("classifier_source", "TEXT"),
    ("btc_price_after_30s", "REAL"),
    ("btc_price_after_60s", "REAL"),
    ("btc_price_after_300s", "REAL"),
    ("optimal_lean", "REAL"),
]

TICK_COLUMN_DEFINITIONS = dict(TICK_COLUMNS)
MARKET_SETTLEMENT_COLUMNS = {
    "market_id": "TEXT PRIMARY KEY",
    "settled_at": "REAL",
    "winning_side": "TEXT",
    "payout_yes": "REAL",
    "payout_no": "REAL",
    "num_yes_shares": "REAL",
    "num_no_shares": "REAL",
    "realized_pnl": "REAL",
    "source": "TEXT",
    "mode": "INTEGER",
}
TICK_BACKFILL_COLUMNS = {
    "btc_price_after_30s",
    "btc_price_after_60s",
    "btc_price_after_300s",
    "optimal_lean",
}
TICK_LOG_COLUMNS = [
    name for name, _ in TICK_COLUMNS
    if name not in TICK_BACKFILL_COLUMNS
]


def ensure_tick_columns(conn):
    """Backfill schema changes safely for existing SQLite databases."""
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(ticks)").fetchall()
    }

    for column, definition in TICK_COLUMN_DEFINITIONS.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE ticks ADD COLUMN {column} {definition}")

    conn.commit()


def ensure_table_columns(conn, table_name: str, column_definitions: dict[str, str]):
    existing_columns = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }

    for column, definition in column_definitions.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}")

    conn.commit()


def init_db(path=None):
    """Create database and tables if they don't exist."""
    path = path or config.LOG_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    c = conn.cursor()

    tick_columns_sql = ",\n            ".join(
        f"{name} {definition}" for name, definition in TICK_COLUMNS
    )

    # ── Main tick log: every signal reading + decision ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {tick_columns_sql}
        )
    """.format(tick_columns_sql=tick_columns_sql))

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

    # ── Market settlement log ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_settlements (
            market_id TEXT PRIMARY KEY,
            settled_at REAL,
            winning_side TEXT,
            payout_yes REAL,
            payout_no REAL,
            num_yes_shares REAL,
            num_no_shares REAL,
            realized_pnl REAL,
            source TEXT,
            mode INTEGER
        )
    """)

    ensure_table_columns(conn, "ticks", TICK_COLUMN_DEFINITIONS)
    ensure_table_columns(conn, "market_settlements", MARKET_SETTLEMENT_COLUMNS)

    conn.commit()
    return conn


def log_tick(conn, data: dict):
    """Insert one tick row. Pass a dict with column names as keys."""
    values = [data.get(column) for column in TICK_LOG_COLUMNS]
    placeholders = ",".join(["?"] * len(TICK_LOG_COLUMNS))
    col_str = ",".join(TICK_LOG_COLUMNS)
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
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


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
    feature_columns = model_feature_names()
    feature_sql = ",\n            ".join(feature_columns)
    query = """
        SELECT
            timestamp,
            {feature_sql},
            optimal_lean
        FROM ticks
        WHERE btc_price_after_60s IS NOT NULL
          AND optimal_lean IS NOT NULL
          AND COALESCE(mid_source, 'polymarket') = 'polymarket'
        ORDER BY timestamp ASC
    """.format(feature_sql=feature_sql)
    rows = conn.execute(query).fetchall()
    columns = ["timestamp", *feature_columns, "optimal_lean"]
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


def get_settlement_summary(conn, start_ts=None, end_ts=None):
    """Aggregate realized performance from settled markets only."""
    where_clauses = ["COALESCE(source, '') != 'synthetic-flat-unwind'"]
    params = []
    if start_ts is not None:
        where_clauses.append("settled_at >= ?")
        params.append(start_ts)
    if end_ts is not None:
        where_clauses.append("settled_at < ?")
        params.append(end_ts)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS settled_count,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN realized_pnl = 0 THEN 1 ELSE 0 END) AS pushes,
            COALESCE(SUM(realized_pnl), 0.0) AS total_pnl
        FROM market_settlements
        {where_sql}
        """,
        params,
    ).fetchone()

    settled_count = int(row[0] or 0)
    wins = int(row[1] or 0)
    losses = int(row[2] or 0)
    pushes = int(row[3] or 0)
    decided_count = wins + losses
    total_pnl = float(row[4] or 0.0)

    return {
        "settled_count": settled_count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "decided_count": decided_count,
        "win_rate": (wins / decided_count) if decided_count else None,
        "total_pnl": round(total_pnl, 4),
    }


def is_market_settled(conn, market_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM market_settlements WHERE market_id=? LIMIT 1",
        (market_id,),
    ).fetchone()
    return bool(row)


def record_market_settlement(conn, data: dict):
    conn.execute("""
        INSERT OR REPLACE INTO market_settlements (
            market_id, settled_at, winning_side, payout_yes, payout_no,
            num_yes_shares, num_no_shares, realized_pnl, source, mode
        ) VALUES (
            :market_id, :settled_at, :winning_side, :payout_yes, :payout_no,
            :num_yes_shares, :num_no_shares, :realized_pnl, :source, :mode
        )
    """, data)
    conn.commit()


def get_win_rate_summary(conn):
    rows = conn.execute(
        """
        SELECT
            ms.market_id,
            ms.realized_pnl,
            COALESCE(
                ms.mode,
                (
                    SELECT pt.mode
                    FROM paper_trades pt
                    WHERE pt.market_id = ms.market_id
                      AND pt.mode IS NOT NULL
                    GROUP BY pt.mode
                    ORDER BY COUNT(*) DESC, pt.mode ASC
                    LIMIT 1
                )
            ) AS resolved_mode
        FROM market_settlements ms
        WHERE COALESCE(ms.source, '') != 'synthetic-flat-unwind'
        """
    ).fetchall()

    mode_stats = {
        mode: {
            "mode": mode,
            "label": {1: "Quiet", 2: "Lean", 3: "Event", 4: "Arb"}.get(mode, "?"),
            "settled_count": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "decided_count": 0,
            "win_rate": None,
            "total_pnl": 0.0,
        }
        for mode in range(1, 5)
    }

    overall = {
        "settled_count": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "decided_count": 0,
        "win_rate": None,
        "total_pnl": 0.0,
    }

    for _, realized_pnl, resolved_mode in rows:
        pnl = float(realized_pnl or 0.0)
        overall["settled_count"] += 1
        overall["total_pnl"] += pnl

        bucket = None
        if resolved_mode in mode_stats:
            bucket = mode_stats[resolved_mode]
            bucket["settled_count"] += 1
            bucket["total_pnl"] += pnl

        if pnl > 0:
            outcome = "wins"
        elif pnl < 0:
            outcome = "losses"
        else:
            outcome = "pushes"

        overall[outcome] += 1
        if bucket is not None:
            bucket[outcome] += 1

    overall["decided_count"] = overall["wins"] + overall["losses"]
    if overall["decided_count"] > 0:
        overall["win_rate"] = round(overall["wins"] / overall["decided_count"], 4)
    overall["total_pnl"] = round(overall["total_pnl"], 4)

    for stats in mode_stats.values():
        stats["decided_count"] = stats["wins"] + stats["losses"]
        if stats["decided_count"] > 0:
            stats["win_rate"] = round(stats["wins"] / stats["decided_count"], 4)
        stats["total_pnl"] = round(stats["total_pnl"], 4)

    ranked_modes = [
        stats for stats in mode_stats.values()
        if stats["decided_count"] > 0
    ]
    ranked_modes.sort(
        key=lambda stats: (
            stats["win_rate"],
            stats["decided_count"],
            stats["total_pnl"],
        ),
        reverse=True,
    )

    return {
        "overall": overall,
        "by_mode": list(mode_stats.values()),
        "best_mode": ranked_modes[0] if ranked_modes else None,
    }


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
