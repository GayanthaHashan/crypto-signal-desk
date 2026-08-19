"""
Pulls OHLCV (candlestick) data from Binance's PUBLIC REST API.
No API key needed for this — it's read-only market data, so this endpoint
never touches your account or funds.

This is intentionally the ONLY place in the app that talks to Binance for
market data, so if you ever want to swap exchanges (Bybit, Coinbase, etc.)
you only need to edit this one file.
"""
import httpx
import pandas as pd
from datetime import datetime

BASE_URL = "https://api.binance.com"

# Binance interval strings: 1m 5m 15m 1h 4h 1d 1w
VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d", "1w"}


class BinanceClientError(Exception):
    pass


async def get_klines(symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
    """
    Fetch candlestick data for a symbol, e.g. symbol='BTCUSDT'.
    Returns a DataFrame with columns: open_time, open, high, low, close, volume
    """
    if interval not in VALID_INTERVALS:
        raise BinanceClientError(f"interval must be one of {VALID_INTERVALS}")

    symbol = symbol.upper().replace("/", "")
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{BASE_URL}/api/v3/klines", params=params)

    if resp.status_code != 200:
        raise BinanceClientError(
            f"Binance API error {resp.status_code}: {resp.text[:200]}"
        )

    raw = resp.json()
    if not raw:
        raise BinanceClientError(f"No data returned for {symbol} ({interval})")

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["open_time", "open", "high", "low", "close", "volume"]]


async def get_current_price(symbol: str) -> float:
    symbol = symbol.upper().replace("/", "")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{BASE_URL}/api/v3/ticker/price", params={"symbol": symbol})
    if resp.status_code != 200:
        raise BinanceClientError(f"Binance API error {resp.status_code}: {resp.text[:200]}")
    return float(resp.json()["price"])
