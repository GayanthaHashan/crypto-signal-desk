# Signal Desk — Crypto Analysis & Paper Trading Bot

A cloud-deployable tool that pulls live chart data from Binance's public API,
runs technical analysis (RSI, MACD, EMAs, Fibonacci retracement, candlestick
patterns) plus free news-sentiment scanning, combines it into a single
weighted signal, and lets that signal trade automatically — **but only
against a simulated ("paper") account, never real money**, unless you
deliberately wire up live execution later (see "Going live" below).

## Read this first

This is a rule-based scoring system, not a prediction engine. No AI or
algorithm — mine or a hedge fund's — can reliably predict crypto price
moves. Treat every signal as one input to think about, not an instruction
to follow blindly. Paper trade for weeks or months and look honestly at
the results before ever considering real capital, and even then, only with
strict position limits. This is not financial advice.

## What's new: adaptive weights + daily AI analyst

Two additions on top of the base signal engine:

- **Adaptive weighting** (`adaptive_weights.py`): every signal the engine
  produces gets logged with its raw per-component scores. ~24h later, a
  background job checks what price actually did and grades whether each
  component (RSI, MACD, trend, Fibonacci, candlestick, news) called the
  direction correctly. Once a component has 20+ graded samples, its future
  weight scales up or down based on its actual hit rate — components that
  are basically coin-flips get downweighted automatically. **This is a
  performance-weighted ensemble, not a neural network retraining itself** —
  worth being precise about, since it needs real sample size (weeks of
  data) before the adjustments mean anything.
- **LLM analyst layer** (`llm_analyst.py`): calls the Anthropic API to
  write a short note reasoning over the quantitative signal — naming which
  classic strategy framework the setup resembles (trend-following,
  mean-reversion, breakout, support/resistance confluence) and flagging
  the biggest risk to that read. It only reasons over numbers the engine
  already computed; it can't invent prices or news.
- **Daily scheduler** (`scheduler.py`): runs the full pipeline (signal →
  log → LLM commentary → paper trade if strong enough) once a day at 08:00
  UTC for every symbol on your watchlist, plus a maintenance job every 6h
  that grades old signals and recomputes weights. You add/remove watchlist
  symbols from the dashboard — there's no hardcoded coin list.

**Setup**: copy `.env.example` to `.env` and add a `GEMINI_API_KEY` — Google's
Gemini API has a genuine free tier with no credit card required, and daily
usage here is small enough to stay well within it. Get one at
https://aistudio.google.com/apikey. If you'd rather use Anthropic's API
instead (paid, no free tier), set `ANTHROPIC_API_KEY` and uncomment the
`anthropic` line in `requirements.txt` — Gemini is tried first, Anthropic
is the fallback. Without either key, everything else still works — you
just won't get the written analyst notes.

## What it does

- **Data**: pulls OHLCV candles straight from Binance's public REST API
  (`binance_client.py`) — no API key needed, since it's read-only market data.
- **Indicators**: RSI, MACD, EMA 20/50/200, Bollinger Bands, ATR
  (`indicators.py`), computed with the `ta` library.
- **Fibonacci**: retracement levels from the recent swing high/low, and
  which zone price is currently sitting in.
- **Candlestick patterns**: rule-based detection of engulfing candles,
  hammers, shooting stars, and dojis.
- **News sentiment**: free RSS feeds (CoinDesk, Cointelegraph, Decrypt)
  filtered by coin keyword, scored with VADER sentiment analysis — no paid
  news API needed to start.
- **Signal engine**: combines all of the above into a weighted -100 to
  +100 score with a plain-English explanation for every component
  (`signal_engine.py`). Weights are tunable in one place.
- **Paper trading**: a fully simulated account (starts at $10,000 virtual
  balance, 2% risk per trade) that can open/close positions based on
  signals and track real PnL and win rate — with zero risk to real funds.
- **Dashboard**: single-page UI to run analysis and watch the paper
  account (`static/index.html`).

## Running it locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

## Deploying to the cloud (free tier)

This is built for **Render's free web service tier** ($0/month, sleeps
after inactivity but wakes on request):

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com), New → Web Service → connect the repo.
   Render will auto-detect `render.yaml` and configure everything.
3. Deploy. Your dashboard is live at `https://your-app.onrender.com`.

Free tier notes:
- The included SQLite database resets if the service restarts/redeploys on
  Render's free tier (its disk isn't persistent). Fine for testing; for a
  paper-trading history you actually want to keep, either upgrade to a paid
  instance with a persistent disk, or point `DATABASE_URL` at a free
  Postgres instance (Supabase and Render both offer one).
- Railway and Fly.io are solid free-tier alternatives if you'd rather not
  use Render.

## Extending it

- **Better news data**: swap `news_sentiment.py` to use CryptoPanic or
  NewsAPI's free tier (just needs an API key in a `.env` file) if the RSS
  feeds feel too thin.
- **Scheduled auto-analysis**: `apscheduler` is already in
  `requirements.txt` — wire up a daily job that runs `/api/paper-trade/open`
  for your watchlist automatically and logs the results.
- **Backtesting**: before trusting any tuning, backtest changes against
  months of historical Binance data rather than eyeballing recent paper
  trades — a handful of trades isn't a reliable signal either way.
- **Going live (real money)**: only after you've validated a strategy
  extensively in paper trading. You'd add a `binance_orders.py` module using
  your Binance API key (with **trade-only permissions, never withdrawal
  permissions**), keep the same risk-sizing logic, and add a manual
  confirmation step before any real order — never let it run unattended
  with real funds without hard position/loss limits.

## Project structure

```
app/
  binance_client.py    # market data (public, read-only)
  indicators.py         # RSI, MACD, EMA, Fibonacci, candlestick patterns
  news_sentiment.py      # free RSS + VADER sentiment
  signal_engine.py       # combines everything into one weighted score
  paper_trading.py       # simulated account, trades, PnL
  database.py             # SQLite/Postgres models
  main.py                  # FastAPI app + endpoints
static/index.html          # dashboard
render.yaml                 # free-tier cloud deploy config
```
