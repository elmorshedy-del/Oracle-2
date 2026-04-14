"""
Auto-Tuner — CatBoost training pipeline.

1. Waits for enough labeled data in SQLite
2. Computes optimal_lean retroactively from future BTC prices
3. Trains CatBoost regressor to predict optimal lean
4. Validates on held-out data
5. Deploys model if it beats minimum accuracy threshold
6. Retrains periodically as new data flows in

This replaces the deterministic rules in RegimeClassifier
once the model is good enough.
"""

import time
import json
import logging
import os
import numpy as np

import config
import database

log = logging.getLogger("tuner")


def compute_optimal_leans(conn):
    """
    Retroactively compute what the optimal lean direction would have been
    for each tick, based on what BTC actually did afterward.

    Logic: if BTC went up 0.3% in the next 60 seconds, the optimal lean
    was +0.3 (positive = bullish). Scaled to [-1, +1].

    This is the training label for CatBoost.
    """
    unlabeled = conn.execute("""
        SELECT id, btc_price, btc_price_after_60s
        FROM ticks
        WHERE btc_price_after_60s IS NOT NULL
          AND optimal_lean IS NULL
          AND btc_price > 0
    """).fetchall()

    if not unlabeled:
        return 0

    count = 0
    for tick_id, price_now, price_60s in unlabeled:
        if price_now <= 0:
            continue
        pct_change = (price_60s - price_now) / price_now
        # Scale: 0.5% move = lean of 1.0 (or -1.0)
        optimal = max(-1.0, min(1.0, pct_change / 0.005))
        conn.execute(
            "UPDATE ticks SET optimal_lean=? WHERE id=?",
            (round(optimal, 4), tick_id)
        )
        count += 1

    conn.commit()
    log.info(f"Labeled {count} ticks with optimal_lean")
    return count


def backfill_future_prices(conn, price_history: list):
    """
    Fill in btc_price_after_30s/60s/300s for ticks that are now old enough.

    price_history: list of (timestamp, price) tuples, sorted by time.
    """
    if len(price_history) < 10:
        return

    ticks = database.get_unlabeled_ticks(conn, max_age_seconds=310)

    def find_price_at(target_ts):
        """Binary search-ish: find price closest to target timestamp."""
        best = None
        best_diff = float("inf")
        for ts, price in price_history:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = price
            if ts > target_ts + 5:
                break
        return best if best_diff < 10 else None  # within 10 seconds

    count = 0
    for tick_id, tick_ts, tick_price in ticks:
        p30 = find_price_at(tick_ts + 30)
        p60 = find_price_at(tick_ts + 60)
        p300 = find_price_at(tick_ts + 300)

        if p60 is not None:  # at minimum need 60s price
            database.backfill_future_prices(conn, tick_id, p30, p60, p300)
            count += 1

    if count:
        log.debug(f"Backfilled future prices for {count} ticks")


def should_train(conn) -> bool:
    """Check if we have enough new data to train or retrain."""
    total = conn.execute(
        "SELECT COUNT(*) FROM ticks WHERE optimal_lean IS NOT NULL"
    ).fetchone()[0]

    if total < config.MIN_SAMPLES_TO_TRAIN:
        return False

    last_trained = database.get_last_training_row_count(conn)
    if total - last_trained >= config.RETRAIN_INTERVAL_ROWS:
        return True

    return False


def train_model(conn) -> dict:
    """
    Train CatBoost on collected data.

    Features: all signal values at each tick
    Target: optimal_lean (what the correct lean would have been)

    Returns dict with training results.
    """
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import train_test_split

    rows, columns = database.get_training_data(conn)
    if len(rows) < config.MIN_SAMPLES_TO_TRAIN:
        return {"error": f"Not enough data: {len(rows)} < {config.MIN_SAMPLES_TO_TRAIN}"}

    log.info(f"Training CatBoost on {len(rows)} samples...")

    # Convert to numpy
    data = np.array(rows, dtype=np.float64)

    # Handle NaN/None → fill with 0
    data = np.nan_to_num(data, nan=0.0)

    # Features = all columns except the last (optimal_lean)
    feature_names = columns[:-1]
    X = data[:, :-1]
    y = data[:, -1]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - config.TRAIN_TEST_SPLIT, random_state=42
    )

    # Train
    model = CatBoostRegressor(
        iterations=config.CATBOOST_ITERATIONS,
        depth=config.CATBOOST_DEPTH,
        learning_rate=config.CATBOOST_LEARNING_RATE,
        loss_function="RMSE",
        verbose=False,
    )

    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    test_pool = Pool(X_test, y_test, feature_names=feature_names)

    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Directional accuracy: did the model predict the right direction?
    train_dir_acc = np.mean(np.sign(train_preds) == np.sign(y_train))
    test_dir_acc = np.mean(np.sign(test_preds) == np.sign(y_test))

    # RMSE
    train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))

    # Feature importance
    importance = dict(zip(feature_names, model.get_feature_importance().tolist()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    result = {
        "num_samples": len(rows),
        "train_accuracy": round(train_dir_acc, 4),
        "test_accuracy": round(test_dir_acc, 4),
        "train_rmse": round(train_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "feature_importance": importance,
        "top_features": top_features,
        "deployed": False,
    }

    log.info(f"  Train directional accuracy: {train_dir_acc:.2%}")
    log.info(f"  Test directional accuracy:  {test_dir_acc:.2%}")
    log.info(f"  Test RMSE:                 {test_rmse:.4f}")
    log.info(f"  Top features: {top_features}")

    # Deploy if good enough
    if test_dir_acc >= config.MODEL_MIN_ACCURACY:
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        model.save_model(config.MODEL_PATH)
        result["deployed"] = True
        log.info(f"  Model deployed to {config.MODEL_PATH}")
    else:
        log.info(
            f"  Model not deployed: {test_dir_acc:.2%} < "
            f"{config.MODEL_MIN_ACCURACY:.2%} threshold"
        )

    # Log training run
    conn.execute("""
        INSERT INTO training_log (timestamp, num_samples, train_accuracy,
            test_accuracy, feature_importance, model_deployed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        time.time(), len(rows), train_dir_acc, test_dir_acc,
        json.dumps(importance), 1 if result["deployed"] else 0
    ))
    conn.commit()

    return result


def run_tuning_cycle(conn, price_history: list) -> dict:
    """
    Called periodically from main loop.
    1. Backfill future prices for old ticks
    2. Compute optimal leans
    3. Train if enough data

    Returns status dict.
    """
    status = {"backfilled": 0, "labeled": 0, "trained": False}

    # Step 1: backfill
    backfill_future_prices(conn, price_history)

    # Step 2: label
    labeled = compute_optimal_leans(conn)
    status["labeled"] = labeled

    # Step 3: train if ready
    if should_train(conn):
        result = train_model(conn)
        status["trained"] = True
        status["training_result"] = result

    return status
