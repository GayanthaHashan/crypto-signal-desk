"""
Combines technical indicators + Fibonacci + candlestick patterns + news
sentiment into a single weighted score (-100 to +100) and a plain-English
explanation of WHY.

IMPORTANT HONESTY NOTE: this is a rule-based weighted scoring system, not
a magic prediction engine. It will be wrong often — markets are noisy.
Treat the output as one input into a decision, not a command. The weights
below are a reasonable starting point; they're the main thing you'll want
to tune/backtest over time.
"""
from app.indicators import add_indicators, fibonacci_levels, detect_candlestick_patterns

# starting weights — these get overridden by the adaptive weight engine
# once there's enough graded signal history (see adaptive_weights.py)
DEFAULT_WEIGHTS = {
    "rsi": 20,
    "macd": 20,
    "trend": 20,       # EMA20/50/200 alignment
    "fibonacci": 15,
    "candlestick": 15,
    "news": 10,
}


def _score_rsi(latest) -> tuple[float, str]:
    rsi = latest["rsi_14"]
    if pd_isnan(rsi):
        return 0, "RSI unavailable"
    if rsi < 30:
        return 1.0, f"RSI {rsi:.1f} — oversold, favors a bounce"
    if rsi > 70:
        return -1.0, f"RSI {rsi:.1f} — overbought, favors a pullback"
    if rsi < 45:
        return 0.3, f"RSI {rsi:.1f} — mildly bearish momentum"
    if rsi > 55:
        return -0.3, f"RSI {rsi:.1f} — mildly bullish momentum, but not extreme"
    return 0, f"RSI {rsi:.1f} — neutral"


def _score_macd(latest) -> tuple[float, str]:
    hist = latest["macd_hist"]
    if pd_isnan(hist):
        return 0, "MACD unavailable"
    if hist > 0:
        return min(hist * 5, 1.0), f"MACD histogram positive ({hist:.4f}) — bullish momentum"
    return max(hist * 5, -1.0), f"MACD histogram negative ({hist:.4f}) — bearish momentum"


def _score_trend(latest) -> tuple[float, str]:
    price = latest["close"]
    ema20, ema50, ema200 = latest["ema_20"], latest["ema_50"], latest["ema_200"]
    if pd_isnan(ema200):
        return 0, "Not enough history for the 200-EMA trend read"
    if price > ema20 > ema50 > ema200:
        return 1.0, "Price above EMA20 > EMA50 > EMA200 — clean uptrend"
    if price < ema20 < ema50 < ema200:
        return -1.0, "Price below EMA20 < EMA50 < EMA200 — clean downtrend"
    if price > ema50:
        return 0.3, "Price above EMA50 — mild uptrend bias"
    if price < ema50:
        return -0.3, "Price below EMA50 — mild downtrend bias"
    return 0, "No clear trend alignment"


def _score_fibonacci(fib: dict) -> tuple[float, str]:
    level = fib["nearest_level"]
    price = fib["current_price"]
    # price near the 0.618/0.5 "golden zone" during a retracement is a classic bounce area
    if level in ("0.5", "0.618"):
        return 0.6, f"Price sitting near the {level} Fibonacci retracement — a classic reaction zone"
    if level == "0.786":
        return 0.3, "Price deep in the 0.786 retracement zone — possible exhaustion area"
    if level in ("0.0", "1.0"):
        return 0.0, f"Price near the {level} swing extreme — no retracement edge yet"
    return 0.1, f"Price near the {level} Fibonacci level"


def _score_candles(patterns: list[dict]) -> tuple[float, str]:
    if not patterns:
        return 0, "No notable candlestick pattern in recent candles"
    recent = patterns[-1]
    bias_score = {"bullish": 0.7, "bearish": -0.7, "neutral": 0}
    return bias_score.get(recent["bias"], 0), f"{recent['pattern']} detected — {recent['bias']} bias"


def _score_news(news: dict) -> tuple[float, str]:
    avg = news["average_sentiment"]
    if news["articles_found"] == 0:
        return 0, "No recent relevant news found"
    return max(min(avg, 1.0), -1.0), (
        f"News sentiment {news['sentiment_label']} across {news['articles_found']} "
        f"recent headlines (avg score {avg})"
    )


def pd_isnan(val) -> bool:
    try:
        return val != val  # NaN != NaN
    except Exception:
        return True


async def build_signal(symbol: str, df, news: dict, weights: dict = None) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    df = add_indicators(df)
    latest = df.iloc[-1]
    fib = fibonacci_levels(df)
    patterns = detect_candlestick_patterns(df)

    components = {
        "rsi": _score_rsi(latest),
        "macd": _score_macd(latest),
        "trend": _score_trend(latest),
        "fibonacci": _score_fibonacci(fib),
        "candlestick": _score_candles(patterns),
        "news": _score_news(news),
    }

    weighted_total = sum(components[k][0] * weights[k] for k in weights)
    max_possible = sum(weights.values())
    final_score = round((weighted_total / max_possible) * 100, 1)  # -100 to +100

    if final_score >= 25:
        direction = "LONG"
    elif final_score <= -25:
        direction = "SHORT"
    else:
        direction = "NEUTRAL / NO TRADE"

    confidence = round(min(abs(final_score), 100), 1)

    reasoning = [f"{k.upper()}: {components[k][1]}" for k in weights]
    raw_component_scores = {k: components[k][0] for k in weights}

    return {
        "symbol": symbol,
        "price": float(latest["close"]),
        "score": final_score,
        "direction": direction,
        "confidence": confidence,
        "reasoning": reasoning,
        "component_scores": raw_component_scores,
        "weights_used": weights,
        "fibonacci": fib,
        "candlestick_patterns": patterns,
        "news_summary": {
            "label": news["sentiment_label"],
            "avg_sentiment": news["average_sentiment"],
            "articles_found": news["articles_found"],
        },
        "disclaimer": (
            "This is a rule-based technical/sentiment score, not a guarantee. "
            "Treat it as one input, backtest before trusting it, and never "
            "risk more than you can afford to lose."
        ),
    }
