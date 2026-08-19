"""
Technical analysis: trend/momentum indicators, Fibonacci retracement levels,
and basic candlestick pattern detection — all computed from OHLCV data
already fetched by binance_client.py (no extra network calls here).
"""
import pandas as pd
import numpy as np
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, EMA20/50/200, Bollinger Bands, ATR to the dataframe."""
    df = df.copy()
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    return df


def fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Computes Fibonacci retracement levels from the swing high/low over the
    last `lookback` candles. Returns levels + which zone the current price sits in.
    """
    window = df.tail(lookback)
    swing_high = window["high"].max()
    swing_low = window["low"].min()
    diff = swing_high - swing_low

    levels = {
        "0.0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0": swing_low,
    }

    current_price = df["close"].iloc[-1]
    # find the nearest fib level to current price
    nearest = min(levels.items(), key=lambda kv: abs(kv[1] - current_price))

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels": levels,
        "current_price": current_price,
        "nearest_level": nearest[0],
        "nearest_level_price": nearest[1],
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    """
    Lightweight rule-based candlestick pattern detection on the last few
    candles (no TA-Lib dependency needed — pure pandas rules).
    Returns a list of {pattern, candle_index, bias} dicts for whatever
    patterns were found in the most recent candles.
    """
    patterns = []
    if len(df) < 3:
        return patterns

    d = df.tail(5).reset_index(drop=True)

    for i in range(1, len(d)):
        o, h, l, c = d.loc[i, ["open", "high", "low", "close"]]
        po, ph, pl, pc = d.loc[i - 1, ["open", "high", "low", "close"]]
        body = abs(c - o)
        candle_range = h - l if h != l else 1e-9
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Bullish / Bearish Engulfing
        if c > o and pc < po and c > po and o < pc:
            patterns.append({"pattern": "Bullish Engulfing", "index": i, "bias": "bullish"})
        if c < o and pc > po and c < po and o > pc:
            patterns.append({"pattern": "Bearish Engulfing", "index": i, "bias": "bearish"})

        # Hammer (small body, long lower wick, little upper wick)
        if body / candle_range < 0.35 and lower_wick > body * 2 and upper_wick < body:
            patterns.append({"pattern": "Hammer", "index": i, "bias": "bullish"})

        # Shooting Star (small body, long upper wick)
        if body / candle_range < 0.35 and upper_wick > body * 2 and lower_wick < body:
            patterns.append({"pattern": "Shooting Star", "index": i, "bias": "bearish"})

        # Doji (very small body relative to range)
        if body / candle_range < 0.1:
            patterns.append({"pattern": "Doji", "index": i, "bias": "neutral"})

    return patterns
