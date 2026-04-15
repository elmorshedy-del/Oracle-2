"""
Shared CatBoost feature schema for training and inference.

Only values known at decision time belong here. Future-price columns stay out
of the model inputs and are reserved for labeling and evaluation.
"""

FEATURE_COLUMNS = [
    # Existing core signals
    "btc_price",
    "btc_momentum",
    "btc_direction",
    "btc_velocity",
    "poly_yes_best_bid",
    "poly_yes_best_ask",
    "poly_no_best_bid",
    "poly_no_best_ask",
    "poly_mid_price",
    "poly_spread",
    "poly_orderbook_imbalance",
    "poly_volume_24h",
    "poly_seconds_remaining",
    "kalshi_yes_price",
    "kalshi_no_price",
    "cross_platform_spread",
    # Volatility regime
    "btc_volatility_15",
    "btc_volatility_60",
    "btc_vol_ratio",
    # Price structure
    "dist_from_high",
    "dist_from_low",
    # Momentum structure
    "momentum_5s",
    "momentum_30s",
    "momentum_divergence",
    # Time patterns
    "hour_of_day",
    "day_of_week",
    "is_us_market_hours",
    # Cross-venue
    "btc_funding_rate",
]


def model_feature_names() -> list[str]:
    return list(FEATURE_COLUMNS)


def feature_vector_from_signals(signals: dict) -> list[float]:
    return [float(signals.get(name, 0) or 0) for name in FEATURE_COLUMNS]
