"""
Configuration for Polymarket Paper Trading Bot
All parameters in one place. Tune these as you collect data.
"""

# ═══════════════════════════════════════════
#  API KEYS (set via environment variables!)
#  NEVER hardcode keys in this file.
#  Create a .env file or export in your shell:
#    export NEWSAPI_KEY="your_key_here"
#    export ANTHROPIC_API_KEY="your_key_here"
# ═══════════════════════════════════════════
import os
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ═══════════════════════════════════════════
#  GENERAL
# ═══════════════════════════════════════════
PAPER_MODE = True  # NEVER set False until you've validated for weeks
STARTING_CAPITAL = 5000.0  # USD
POLL_INTERVAL_SEC = 2  # Main loop frequency
LOG_DB_PATH = "data/trades.db"
MODEL_PATH = "data/catboost_model.cbm"

# ═══════════════════════════════════════════
#  BINANCE FEED (Signal 1: Market Data)
# ═══════════════════════════════════════════
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_MOMENTUM_WINDOW = 15  # seconds of price history for momentum calc
BINANCE_DIRECTION_THRESHOLD = 0.0015  # 0.15% move = directional signal

# ═══════════════════════════════════════════
#  POLYMARKET FEED (Orderbook Data)
# ═══════════════════════════════════════════
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
MARKET_SLUGS = [
    "will-bitcoin-go-up-or-down-in-the-next-5-minutes",
    "will-bitcoin-go-up-or-down-in-the-next-15-minutes",
]
# Fallback: search by tag
MARKET_TAGS = ["btc", "bitcoin"]
MARKET_REFRESH_INTERVAL = 30  # seconds between checking for new markets (short-term rotate fast)

# ═══════════════════════════════════════════
#  NEWS / LLM SIGNAL (Signal 2: Event Detection)
# ═══════════════════════════════════════════
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWS_POLL_INTERVAL = 300  # seconds between news checks (5 min = free tier safe)
NEWS_KEYWORDS = [
    "bitcoin", "btc", "crypto", "federal reserve", "interest rate",
    "sec crypto", "ethereum", "binance", "coinbase", "regulation",
    "tariff", "inflation", "cpi", "employment", "gdp",
]
NEWS_MAX_AGE_MINUTES = 30  # ignore headlines older than this
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 300
LLM_EDGE_THRESHOLD = 0.08  # 8% divergence from market = tradeable
LLM_CONFIDENCE_THRESHOLD = 0.6  # minimum LLM confidence to trigger Mode 3

# ═══════════════════════════════════════════
#  KALSHI FEED (Signal 3: Cross-Platform)
# ═══════════════════════════════════════════
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES_TICKER = "KXBTC"  # BTC hourly markets
KALSHI_POLL_INTERVAL = 5  # seconds

# ═══════════════════════════════════════════
#  REGIME CLASSIFIER (Deterministic Rules)
# ═══════════════════════════════════════════

# Mode 2: Informed Lean thresholds
LEAN_BINANCE_THRESHOLD = 0.0010  # 0.10% BTC move triggers lean
LEAN_CONFIDENCE_SCALE = 5.0  # multiplier: move_pct * scale = confidence
LEAN_MAX_CONFIDENCE = 0.8

# Mode 3: News Event thresholds (placeholder until LLM integration)
# For now, triggers on extreme Binance moves as proxy for "event"
NEWS_PROXY_THRESHOLD = 0.005  # 0.50% BTC move in <10s = "event"

# Mode 4: Cross-platform Arb thresholds
ARB_MIN_SPREAD = 0.03  # 3 cents minimum combined edge
ARB_MAX_COMBINED_COST = 0.97  # buy both sides for < $0.97

# ═══════════════════════════════════════════
#  PAPER TRADER (Simulated Execution)
# ═══════════════════════════════════════════
DEFAULT_ORDER_SIZE = 50  # shares per side
SPREAD_BPS = 200  # 2% spread in basis points (each side 1% from mid)
MAX_INVENTORY_PER_SIDE = 500  # max shares held on one side
MAX_TOTAL_EXPOSURE = 1000.0  # max USD exposure across all positions
DAILY_DRAWDOWN_LIMIT = 0.05  # 5% of capital = hard stop
FILL_PROBABILITY = 0.3  # probability a paper limit order fills per cycle
ORDER_TTL_SEC = 30  # how long resting paper quotes stay live before expiring
PASSIVE_FILL_RATIO = 0.15  # odds of a resting maker order filling without a full cross
PASSIVE_FILL_DISTANCE_BPS = 100  # resting quote can fill if within 1% of the reference price
MAKER_REBATE_BPS = 10  # ~0.1% maker rebate (estimate)
TAKER_FEE_MAX_BPS = 156  # 1.56% max taker fee at 50% probability
ORDER_MAX_AGE_SEC = 30  # orders expire after this many seconds
MIN_TRADABLE_SPREAD = 0.20  # skip markets with spread > 20¢ (untradable book)

# ═══════════════════════════════════════════
#  SETTLEMENT (Market Resolution)
# ═══════════════════════════════════════════
SETTLEMENT_PAYOUT = 1.0  # winning side pays $1.00 per share
SETTLEMENT_CHECK_INTERVAL_SEC = 30
SETTLEMENT_WINNER_MIN_PRICE = 0.99
SETTLEMENT_LOSER_MAX_PRICE = 0.01

# ═══════════════════════════════════════════
#  AUTO TUNER (CatBoost Training)
# ═══════════════════════════════════════════
MIN_SAMPLES_TO_TRAIN = 5000  # minimum rows before first training
RETRAIN_INTERVAL_ROWS = 2000  # retrain every N new rows
TRAIN_TEST_SPLIT = 0.8
TRAIN_VALIDATION_GAP_SEC = 60  # keep 60s between train/test to avoid overlap
CATBOOST_ITERATIONS = 500
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.05
USE_MODEL_IF_AVAILABLE = True  # switch from rules to model when ready
MODEL_MIN_ACCURACY = 0.55  # model must beat this to replace rules
