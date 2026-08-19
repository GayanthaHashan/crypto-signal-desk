"""
This is the 'self-improving' part of the bot, and it's worth being precise
about what it actually does: it does NOT retrain a neural network. It keeps
a track record of every signal component (RSI, MACD, trend, Fibonacci,
candlestick, news) and, ~24h later, checks whether that component's
individual bias (bullish/bearish) matched which way price actually moved.
Over time, components with a better hit rate get more weight in future
signals, and components that are basically coin-flips get downweighted.

This is a legitimate, honest form of adaptation — a simple performance-
weighted ensemble — not a claim of deep learning. It also needs real
sample size (dozens of graded signals per component minimum) before the
adjustments mean anything; with too few samples it defaults back to the
original static weights so it doesn't overreact to a handful of trades.
"""
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from app.database import SignalRecord, ComponentWeight
from app.signal_engine import DEFAULT_WEIGHTS
from app.binance_client import get_current_price

MIN_SAMPLES_TO_ADAPT = 20          # below this, a component keeps its default weight
GRADE_AFTER_HOURS = 24
FLAT_THRESHOLD_PCT = 0.15          # price move smaller than this counts as FLAT, not UP/DOWN


def get_weights(db: Session) -> dict:
    """Returns the current weight dict to use for scoring — adaptive where enough data exists, default otherwise."""
    stored = {cw.component: cw for cw in db.query(ComponentWeight).all()}
    weights = {}
    for component, default_w in DEFAULT_WEIGHTS.items():
        cw = stored.get(component)
        if cw and cw.sample_size >= MIN_SAMPLES_TO_ADAPT:
            weights[component] = cw.weight
        else:
            weights[component] = default_w
    return weights


def log_signal(db: Session, symbol: str, price: float, direction: str,
                score: float, component_scores: dict) -> SignalRecord:
    record = SignalRecord(
        symbol=symbol,
        price_at_signal=price,
        direction=direction,
        final_score=score,
        component_scores=component_scores,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


async def grade_pending_signals(db: Session) -> int:
    """
    Finds signals old enough to grade (>= GRADE_AFTER_HOURS), fetches the
    current price, and marks whether price actually went UP/DOWN/FLAT.
    Returns how many were graded.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=GRADE_AFTER_HOURS)
    pending = db.query(SignalRecord).filter(
        SignalRecord.graded == "PENDING",
        SignalRecord.created_at <= cutoff,
    ).all()

    graded_count = 0
    for record in pending:
        try:
            current_price = await get_current_price(record.symbol)
        except Exception:
            continue  # skip, try again next run

        pct_change = (current_price - record.price_at_signal) / record.price_at_signal * 100
        if pct_change > FLAT_THRESHOLD_PCT:
            actual = "UP"
        elif pct_change < -FLAT_THRESHOLD_PCT:
            actual = "DOWN"
        else:
            actual = "FLAT"

        record.outcome_price = current_price
        record.actual_direction = actual
        record.graded = "GRADED"
        record.graded_at = datetime.now(timezone.utc)
        graded_count += 1

    db.commit()
    return graded_count


def recompute_weights(db: Session) -> dict:
    """
    Recalculates each component's weight from its historical hit rate
    across all GRADED signals. A component is 'correct' if its bias sign
    matched the actual direction (positive score + UP, or negative + DOWN).
    FLAT outcomes are excluded — no clear right answer to score against.
    """
    graded = db.query(SignalRecord).filter(SignalRecord.graded == "GRADED",
                                            SignalRecord.actual_direction != "FLAT").all()

    correct = {c: 0 for c in DEFAULT_WEIGHTS}
    total = {c: 0 for c in DEFAULT_WEIGHTS}

    for record in graded:
        for component, comp_score in (record.component_scores or {}).items():
            if component not in DEFAULT_WEIGHTS or comp_score == 0:
                continue
            predicted = "UP" if comp_score > 0 else "DOWN"
            total[component] += 1
            if predicted == record.actual_direction:
                correct[component] += 1

    results = {}
    for component, default_w in DEFAULT_WEIGHTS.items():
        n = total[component]
        accuracy = (correct[component] / n * 100) if n else None

        if n >= MIN_SAMPLES_TO_ADAPT:
            # accuracy of 50% (coin flip) maps back to roughly the default weight;
            # accuracy above/below 50% scales weight up/down, clamped to a sane range
            scale = max(0.4, min(1.8, accuracy / 50))
            new_weight = round(default_w * scale, 2)
        else:
            new_weight = default_w

        cw = db.query(ComponentWeight).filter_by(component=component).first()
        if not cw:
            cw = ComponentWeight(component=component, weight=new_weight)
            db.add(cw)
        cw.weight = new_weight
        cw.accuracy_pct = round(accuracy, 1) if accuracy is not None else None
        cw.sample_size = n
        cw.updated_at = datetime.now(timezone.utc)

        results[component] = {"weight": new_weight, "accuracy_pct": cw.accuracy_pct, "sample_size": n}

    db.commit()
    return results
