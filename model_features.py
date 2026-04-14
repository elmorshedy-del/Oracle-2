"""
Shared CatBoost feature schema for training and inference.

Only values known at decision time belong here. Future-price columns stay out
of the model inputs and are reserved for labeling/evaluation only.
"""

MODEL_FEATURE_COLUMNS = [
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
]


def model_feature_names() -> list[str]:
    return list(MODEL_FEATURE_COLUMNS)


def feature_vector_from_signals(signals: dict) -> list[float]:
    return [float(signals.get(name, 0) or 0) for name in MODEL_FEATURE_COLUMNS]
