"""
Engine — Regime Classifier + Paper Trader + Risk Manager

The regime classifier decides which mode is active.
The paper trader simulates order placement and fills.
The risk manager enforces position limits and drawdown caps.
"""

import time
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import config
from model_features import feature_vector_from_signals, model_feature_names

log = logging.getLogger("engine")


# ═══════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════

@dataclass
class PaperOrder:
    id: int
    timestamp: float
    market_id: str
    side: str          # 'YES' or 'NO'
    order_type: str    # 'BID' or 'ASK'
    price: float
    size: float
    filled: bool = False
    fill_price: float = 0.0
    fill_timestamp: float = 0.0
    mode: int = 1
    lean_direction: float = 0.0
    lean_confidence: float = 0.0
    db_trade_id: Optional[int] = None


@dataclass
class Position:
    market_id: str
    yes_shares: float = 0.0
    no_shares: float = 0.0
    yes_avg_cost: float = 0.0
    no_avg_cost: float = 0.0
    total_cost: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class RegimeDecision:
    mode: int                  # 1, 2, 3, or 4
    lean_direction: float      # -1.0 (bearish) to +1.0 (bullish)
    lean_confidence: float     # 0.0 to 1.0
    source: str                # 'rules' or 'model'
    reason: str                # human-readable explanation


# ═══════════════════════════════════════════════════════
#  REGIME CLASSIFIER (Deterministic Rules)
# ═══════════════════════════════════════════════════════

class RegimeClassifier:
    """
    Deterministic rules that classify current market state into
    one of 4 modes. Will be replaced by CatBoost once enough
    training data is collected.
    """

    def __init__(self):
        self._catboost_model = None
        self._use_model = False
        self._model_prediction_ok = False
        self._model_error = ""

    def load_model(self, model_path):
        """Load trained CatBoost model to replace rules."""
        try:
            from catboost import CatBoostRegressor
            self._catboost_model = CatBoostRegressor()
            self._catboost_model.load_model(model_path)
            trained_features = list(getattr(self._catboost_model, "feature_names_", []) or [])
            expected_features = model_feature_names()
            if trained_features and trained_features != expected_features:
                self._catboost_model = None
                self._use_model = False
                self._model_prediction_ok = False
                self._model_error = (
                    "feature schema mismatch between trained model and live signals"
                )
                log.warning(
                    "Refusing CatBoost model with incompatible features. "
                    f"expected={expected_features}, got={trained_features}"
                )
                return
            self._use_model = True
            self._model_prediction_ok = False
            self._model_error = ""
            log.info("CatBoost model loaded — switching from rules to model")
        except Exception as e:
            log.warning(f"Failed to load model: {e}. Staying with rules.")
            self._catboost_model = None
            self._use_model = False
            self._model_prediction_ok = False
            self._model_error = str(e)

    def runtime_source(self) -> str:
        if self._use_model and self._model_prediction_ok:
            return "model"
        return "rules"

    def classify(self, signals: dict) -> RegimeDecision:
        """
        Main entry point: given all current signals, decide the mode.

        signals dict expected keys:
            btc_momentum, btc_direction, btc_velocity,
            poly_spread, poly_orderbook_imbalance, poly_seconds_remaining,
            cross_platform_spread, short_momentum
        """
        if self._use_model and self._catboost_model:
            return self._classify_model(signals)
        return self._classify_rules(signals)

    def _classify_rules(self, s: dict) -> RegimeDecision:
        """Deterministic rule-based classification."""

        cross_spread = s.get("cross_platform_spread") or 0
        btc_momentum = s.get("btc_momentum") or 0
        btc_direction = s.get("btc_direction") or 0
        btc_velocity = s.get("btc_velocity") or 0
        short_momentum = s.get("short_momentum") or 0
        orderbook_imbalance = s.get("poly_orderbook_imbalance") or 0
        seconds_remaining = s.get("poly_seconds_remaining") or 999

        # News/LLM signal
        news_active = s.get("news_active", False)
        news_direction = s.get("news_direction", 0)
        news_edge = s.get("news_edge", 0)
        news_confidence = s.get("news_confidence", 0)

        # ── Mode 4: Cross-platform arb ──
        # Highest priority — if arb exists, take it
        if cross_spread >= config.ARB_MIN_SPREAD:
            return RegimeDecision(
                mode=4,
                lean_direction=1.0 if cross_spread > 0 else -1.0,
                lean_confidence=min(cross_spread / 0.10, 1.0),
                source="rules",
                reason=f"Cross-platform arb detected: spread={cross_spread:.4f}"
            )

        # ── Mode 3: News/Event ──
        # Priority A: Real LLM news signal
        if news_active and abs(news_edge) > 0:
            return RegimeDecision(
                mode=3,
                lean_direction=float(news_direction),
                lean_confidence=news_confidence,
                source="rules",
                reason=f"LLM news signal: edge={news_edge:+.2%}, "
                       f"conf={news_confidence:.2f}"
            )

        # Priority B: Extreme BTC move as event proxy (fallback)
        abs_short = abs(short_momentum)
        if abs_short >= config.NEWS_PROXY_THRESHOLD:
            direction = 1.0 if short_momentum > 0 else -1.0
            return RegimeDecision(
                mode=3,
                lean_direction=direction,
                lean_confidence=min(abs_short / 0.01, 1.0),
                source="rules",
                reason=f"Event proxy: {short_momentum:.4%} move in <10s"
            )

        # ── Mode 2: Informed lean ──
        # Moderate directional signal from the BTC price feed
        abs_momentum = abs(btc_momentum)
        if abs_momentum >= config.LEAN_BINANCE_THRESHOLD:
            direction = 1.0 if btc_momentum > 0 else -1.0
            confidence = min(abs_momentum * config.LEAN_CONFIDENCE_SCALE, config.LEAN_MAX_CONFIDENCE)

            # Boost confidence if orderbook imbalance confirms direction
            if (direction > 0 and orderbook_imbalance > 0.2) or \
               (direction < 0 and orderbook_imbalance < -0.2):
                confidence = min(confidence * 1.3, config.LEAN_MAX_CONFIDENCE)

            return RegimeDecision(
                mode=2,
                lean_direction=direction,
                lean_confidence=round(confidence, 3),
                source="rules",
                reason=f"BTC feed lean: momentum={btc_momentum:.4%}, "
                       f"OB imbalance={orderbook_imbalance:.3f}"
            )

        # ── Mode 1: Symmetric market making ──
        return RegimeDecision(
            mode=1,
            lean_direction=0.0,
            lean_confidence=0.0,
            source="rules",
            reason="Quiet market — symmetric quotes"
        )

    def _classify_model(self, s: dict) -> RegimeDecision:
        """CatBoost model-based classification."""
        features = feature_vector_from_signals(s)

        try:
            prediction = self._catboost_model.predict([features])[0]
            self._model_prediction_ok = True
            self._model_error = ""
            lean_dir = max(-1.0, min(1.0, prediction))
            confidence = min(abs(lean_dir), 1.0)

            # Map continuous prediction to mode
            if abs(lean_dir) < 0.1:
                mode = 1
            elif abs(lean_dir) < 0.5:
                mode = 2
            else:
                mode = 3

            # Mode 4 override still uses rules (arb is structural)
            cross_spread = s.get("cross_platform_spread") or 0
            if cross_spread >= config.ARB_MIN_SPREAD:
                mode = 4
                lean_dir = 1.0 if cross_spread > 0 else -1.0
                confidence = min(cross_spread / 0.10, 1.0)

            return RegimeDecision(
                mode=mode,
                lean_direction=round(lean_dir, 3),
                lean_confidence=round(confidence, 3),
                source="model",
                reason=f"CatBoost prediction: {lean_dir:.3f}"
            )
        except Exception as e:
            self._model_prediction_ok = False
            self._model_error = str(e)
            log.warning(f"Model prediction failed: {e}. Falling back to rules.")
            return self._classify_rules(s)


# ═══════════════════════════════════════════════════════
#  RISK MANAGER
# ═══════════════════════════════════════════════════════

class RiskManager:
    """
    Enforces position limits, exposure caps, and drawdown limits.
    Sits between the regime classifier and the paper trader.
    """

    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.balance = starting_capital
        self.daily_starting_balance = starting_capital
        self.peak_balance = starting_capital
        self.positions: dict = {}  # market_id → Position
        self.halted = False
        self.halt_reason = ""

    def check_order(self, market_id: str, side: str, size: float, price: float) -> tuple:
        """
        Returns (allowed: bool, adjusted_size: float, reason: str)
        """
        if self.halted:
            return False, 0, f"Trading halted: {self.halt_reason}"

        # Daily drawdown check
        drawdown = (self.daily_starting_balance - self.balance) / self.daily_starting_balance
        if drawdown >= config.DAILY_DRAWDOWN_LIMIT:
            self.halted = True
            self.halt_reason = f"Daily drawdown limit hit: {drawdown:.2%}"
            return False, 0, self.halt_reason

        # Inventory check
        pos = self.positions.get(market_id, Position(market_id=market_id))
        current_side_shares = pos.yes_shares if side == "YES" else pos.no_shares
        if current_side_shares + size > config.MAX_INVENTORY_PER_SIDE:
            adjusted = config.MAX_INVENTORY_PER_SIDE - current_side_shares
            if adjusted <= 0:
                return False, 0, f"Max inventory reached for {side}"
            size = adjusted

        # Total exposure check
        total_exposure = sum(
            p.total_cost for p in self.positions.values()
        )
        order_cost = size * price
        if total_exposure + order_cost > config.MAX_TOTAL_EXPOSURE:
            max_allowed = config.MAX_TOTAL_EXPOSURE - total_exposure
            if max_allowed <= 0:
                return False, 0, "Max total exposure reached"
            size = max_allowed / price

        return True, size, "OK"

    def update_position(self, market_id: str, side: str, shares: float,
                        price: float, is_buy: bool):
        """Update position after a fill and return realized PnL for this action."""
        if market_id not in self.positions:
            self.positions[market_id] = Position(market_id=market_id)

        pos = self.positions[market_id]
        realized_pnl = 0.0

        if is_buy:
            cost = shares * price
            if side == "YES":
                total_shares = pos.yes_shares + shares
                if total_shares > 0:
                    pos.yes_avg_cost = (
                        (pos.yes_avg_cost * pos.yes_shares + cost) / total_shares
                    )
                pos.yes_shares = total_shares
            else:
                total_shares = pos.no_shares + shares
                if total_shares > 0:
                    pos.no_avg_cost = (
                        (pos.no_avg_cost * pos.no_shares + cost) / total_shares
                    )
                pos.no_shares = total_shares
            pos.total_cost += cost
            self.balance -= cost
        else:
            revenue = shares * price
            if side == "YES":
                pnl = (price - pos.yes_avg_cost) * shares
                pos.yes_shares = max(0, pos.yes_shares - shares)
            else:
                pnl = (price - pos.no_avg_cost) * shares
                pos.no_shares = max(0, pos.no_shares - shares)
            pos.realized_pnl += pnl
            pos.total_cost = max(0, pos.total_cost - revenue)
            self.balance += revenue
            realized_pnl = pnl

        # Track peak
        self.peak_balance = max(self.peak_balance, self.balance)
        return realized_pnl

    def apply_rebate(self, rebate: float):
        """Credit maker rebates directly to balance."""
        self.balance += rebate
        self.peak_balance = max(self.peak_balance, self.balance)

    def settle_market(self, market_id: str, payout_yes: float, payout_no: float):
        """Resolve a binary market into cash and realized PnL."""
        pos = self.positions.get(market_id)
        if not pos:
            return None

        yes_shares = pos.yes_shares
        no_shares = pos.no_shares
        yes_cost = yes_shares * pos.yes_avg_cost
        no_cost = no_shares * pos.no_avg_cost
        payout = yes_shares * payout_yes + no_shares * payout_no
        realized_pnl = (payout - yes_cost - no_cost)

        self.balance += payout
        pos.realized_pnl += realized_pnl
        pos.total_cost = 0.0
        pos.yes_shares = 0.0
        pos.no_shares = 0.0
        pos.yes_avg_cost = 0.0
        pos.no_avg_cost = 0.0
        self.peak_balance = max(self.peak_balance, self.balance)
        del self.positions[market_id]

        return {
            "market_id": market_id,
            "num_yes_shares": yes_shares,
            "num_no_shares": no_shares,
            "payout_yes": payout_yes,
            "payout_no": payout_no,
            "payout": round(payout, 4),
            "realized_pnl": round(realized_pnl, 4),
        }

    def reset_daily(self):
        """Reset daily tracking at start of new day."""
        self.daily_starting_balance = self.balance
        self.halted = False
        self.halt_reason = ""

    def get_stats(self) -> dict:
        return {
            "balance": round(self.balance, 2),
            "total_pnl": round(self.balance - self.starting_capital, 2),
            "daily_pnl": round(self.balance - self.daily_starting_balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "num_positions": len(self.positions),
            "halted": self.halted,
        }


# ═══════════════════════════════════════════════════════
#  PAPER TRADER
# ═══════════════════════════════════════════════════════

class PaperTrader:
    """
    Simulates market making on Polymarket. Places hypothetical
    limit orders and simulates fills based on price movement.

    Mode 1: Symmetric bids/asks around mid price
    Mode 2: Asymmetric — tighter on the lean side, wider on other
    Mode 3: One-sided aggressive — only lean side
    Mode 4: Both legs of cross-platform arb
    """

    def __init__(self, risk_manager: RiskManager):
        self.risk = risk_manager
        self.open_orders: List[PaperOrder] = []
        self._next_order_id = 1
        self.total_orders_placed = 0
        self.total_fills = 0

    def generate_orders(self, decision: RegimeDecision, poly_signal: dict) -> List[PaperOrder]:
        """
        Generate paper orders based on the current regime decision
        and polymarket orderbook state.
        """
        self._expire_orders()

        mid = poly_signal.get("poly_mid_price")
        market_id = poly_signal.get("poly_market_id") or "paper_market"
        spread = self._quote_spread()

        # Rotating markets should retire stale quotes before a new book becomes active.
        self._cancel_rotated_out_orders(market_id)

        if mid is None:
            # Use a synthetic mid price for testing when no market data
            mid = 0.50

        orders = []

        if decision.mode == 1:
            orders = self._mode1_symmetric(market_id, mid, spread, decision)

        elif decision.mode == 2:
            orders = self._mode2_lean(market_id, mid, spread, decision)

        elif decision.mode == 3:
            orders = self._mode3_aggressive(market_id, mid, spread, decision)

        elif decision.mode == 4:
            orders = self._mode4_arb(market_id, mid, spread, decision)

        # Risk check each order
        approved = []
        active_keys = {
            (o.market_id, o.side, o.order_type)
            for o in self.open_orders
            if not o.filled
        }
        for order in orders:
            order_key = (order.market_id, order.side, order.order_type)
            if order_key in active_keys:
                continue
            allowed, adj_size, reason = self.risk.check_order(
                order.market_id, order.side, order.size, order.price
            )
            if allowed and adj_size > 0:
                order.size = adj_size
                approved.append(order)
                active_keys.add(order_key)

        self.open_orders.extend(approved)
        self.total_orders_placed += len(approved)
        return approved

    def _mode1_symmetric(self, market_id, mid, spread, decision) -> List[PaperOrder]:
        """Symmetric quotes on both sides."""
        half_spread = max(spread / 2, 0.01)
        return [
            self._make_order(market_id, "YES", "BID", round(mid - half_spread, 3),
                             config.DEFAULT_ORDER_SIZE, decision),
            self._make_order(market_id, "YES", "ASK", round(mid + half_spread, 3),
                             config.DEFAULT_ORDER_SIZE, decision),
            self._make_order(market_id, "NO", "BID", round(1.0 - mid - half_spread, 3),
                             config.DEFAULT_ORDER_SIZE, decision),
            self._make_order(market_id, "NO", "ASK", round(1.0 - mid + half_spread, 3),
                             config.DEFAULT_ORDER_SIZE, decision),
        ]

    def _mode2_lean(self, market_id, mid, spread, decision) -> List[PaperOrder]:
        """Asymmetric: tighter on lean side, wider on other."""
        lean = decision.lean_direction  # +1 = bullish (favor YES)
        conf = decision.lean_confidence
        half_spread = max(spread / 2, 0.01)

        # Tighten spread on favored side, widen on other
        tight = half_spread * (1.0 - conf * 0.5)  # up to 50% tighter
        wide = half_spread * (1.0 + conf * 0.5)   # up to 50% wider

        orders = []
        if lean > 0:
            # Bullish: tight YES bid, wide YES ask
            orders.append(self._make_order(market_id, "YES", "BID",
                          round(mid - tight, 3), config.DEFAULT_ORDER_SIZE, decision))
            orders.append(self._make_order(market_id, "YES", "ASK",
                          round(mid + wide, 3), config.DEFAULT_ORDER_SIZE, decision))
        else:
            # Bearish: tight NO bid, wide NO ask
            no_mid = 1.0 - mid
            orders.append(self._make_order(market_id, "NO", "BID",
                          round(no_mid - tight, 3), config.DEFAULT_ORDER_SIZE, decision))
            orders.append(self._make_order(market_id, "NO", "ASK",
                          round(no_mid + wide, 3), config.DEFAULT_ORDER_SIZE, decision))

        return orders

    def _mode3_aggressive(self, market_id, mid, spread, decision) -> List[PaperOrder]:
        """One-sided aggressive orders during events."""
        lean = decision.lean_direction
        conf = decision.lean_confidence
        size = int(config.DEFAULT_ORDER_SIZE * (1.0 + conf))  # bigger size

        orders = []
        if lean > 0:
            # Bullish: aggressive YES bids at tighter prices
            orders.append(self._make_order(market_id, "YES", "BID",
                          round(mid - 0.005, 3), size, decision))
            orders.append(self._make_order(market_id, "YES", "BID",
                          round(mid - 0.01, 3), size, decision))
        else:
            no_mid = 1.0 - mid
            orders.append(self._make_order(market_id, "NO", "BID",
                          round(no_mid - 0.005, 3), size, decision))
            orders.append(self._make_order(market_id, "NO", "BID",
                          round(no_mid - 0.01, 3), size, decision))

        return orders

    def _mode4_arb(self, market_id, mid, spread, decision) -> List[PaperOrder]:
        """Cross-platform arb — place the Polymarket leg as maker."""
        # In paper mode, simulate buying the cheap side
        lean = decision.lean_direction
        size = config.DEFAULT_ORDER_SIZE

        orders = []
        if lean > 0:
            orders.append(self._make_order(market_id, "YES", "BID",
                          round(mid - 0.01, 3), size, decision))
        else:
            orders.append(self._make_order(market_id, "NO", "BID",
                          round(1.0 - mid - 0.01, 3), size, decision))
        return orders

    def _quote_spread(self) -> float:
        """Use the configured paper spread instead of a raw live-book spread."""
        return max(config.SPREAD_BPS / 10000, 0.02)

    def _expire_orders(self):
        """Drop resting quotes that have gone stale."""
        now = time.time()
        self.open_orders = [
            order for order in self.open_orders
            if order.filled or now - order.timestamp <= config.ORDER_TTL_SEC
        ]

    def _cancel_rotated_out_orders(self, active_market_id: str):
        """Cancel quotes that belong to markets we are no longer pricing."""
        if not active_market_id:
            return
        self.open_orders = [
            order for order in self.open_orders
            if order.filled or order.market_id == active_market_id
        ]

    def simulate_fills(self, current_mid: float, market_id: Optional[str] = None):
        """
        Check if any open paper orders would have filled based on
        current market price movement. Simple model:
        - BID fills if price drops to or below bid price
        - ASK fills if price rises to or above ask price
        Plus randomness to simulate partial fills and queue position.
        """
        self._expire_orders()
        if current_mid is None:
            return []

        fills = []
        remaining = []

        for order in self.open_orders:
            if order.filled:
                remaining.append(order)
                continue

            # Only the active market's mid should be able to fill its own quotes.
            if market_id and order.market_id != market_id:
                remaining.append(order)
                continue

            filled = False

            if order.side == "YES":
                ref_price = current_mid
            else:
                ref_price = 1.0 - current_mid

            if order.order_type == "BID" and ref_price <= order.price:
                # Price dropped to our bid — potential fill
                if random.random() < config.FILL_PROBABILITY:
                    filled = True
            elif order.order_type == "ASK" and ref_price >= order.price:
                # Price rose to our ask — potential fill
                if random.random() < config.FILL_PROBABILITY:
                    filled = True
            else:
                # Resting maker quotes can still get lifted/hit without a full cross.
                distance = abs(ref_price - order.price)
                passive_distance = config.PASSIVE_FILL_DISTANCE_BPS / 10000
                passive_fill_prob = config.FILL_PROBABILITY * config.PASSIVE_FILL_RATIO
                if distance <= passive_distance and random.random() < passive_fill_prob:
                    filled = True

            if filled:
                order.filled = True
                order.fill_price = order.price
                order.fill_timestamp = time.time()
                self.total_fills += 1

                # Update risk manager
                is_buy = order.order_type == "BID"
                realized_pnl = self.risk.update_position(
                    order.market_id, order.side,
                    order.size, order.fill_price, is_buy
                )

                # Calculate paper PnL for this fill
                # Maker rebate
                rebate = order.size * order.price * (config.MAKER_REBATE_BPS / 10000)
                self.risk.apply_rebate(rebate)

                fills.append({
                    "order": order,
                    "rebate": rebate,
                    "realized_pnl": realized_pnl,
                })
                log.info(
                    f"  FILL: {order.order_type} {order.size} {order.side} "
                    f"@ {order.fill_price:.3f} (mode {order.mode}) +${rebate:.2f} rebate"
                )
            else:
                remaining.append(order)

        self.open_orders = remaining
        return fills

    def cancel_all(self, market_id=None, side=None):
        """Cancel open orders, optionally filtered by market/side."""
        before = len(self.open_orders)
        self.open_orders = [
            o for o in self.open_orders
            if not o.filled and
               (market_id is None or o.market_id != market_id) and
               (side is None or o.side != side)
        ]
        cancelled = before - len(self.open_orders)
        if cancelled:
            log.debug(f"  Cancelled {cancelled} orders")

    def _make_order(self, market_id, side, order_type, price, size, decision) -> PaperOrder:
        """Create a new paper order."""
        price = max(0.01, min(0.99, price))
        oid = self._next_order_id
        self._next_order_id += 1
        return PaperOrder(
            id=oid,
            timestamp=time.time(),
            market_id=market_id,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            mode=decision.mode,
            lean_direction=decision.lean_direction,
            lean_confidence=decision.lean_confidence,
        )

    def get_stats(self) -> dict:
        return {
            "open_orders": len(self.open_orders),
            "total_placed": self.total_orders_placed,
            "total_fills": self.total_fills,
            "fill_rate": (
                round(self.total_fills / max(1, self.total_orders_placed), 3)
            ),
        }
