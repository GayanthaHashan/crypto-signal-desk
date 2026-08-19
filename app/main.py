from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import init_db, get_db, WatchlistItem, DailyBrief
from app.binance_client import get_klines, BinanceClientError
from app.news_sentiment import get_news_sentiment
from app.signal_engine import build_signal
from app.llm_analyst import analyze_with_llm
from app import paper_trading, adaptive_weights
from app.scheduler import start_scheduler, daily_analysis_job, maintenance_job

app = FastAPI(title="Crypto Analysis & Paper Trading Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler()


@app.get("/api/analyze/{symbol}")
async def analyze(symbol: str, interval: str = "1h", with_commentary: bool = False,
                   db: Session = Depends(get_db)):
    """
    Full analysis for a symbol e.g. BTCUSDT: fetches Binance candles,
    computes indicators/Fibonacci/candlestick patterns, pulls news
    sentiment, and returns a combined weighted signal — using the current
    adaptive weights (falls back to defaults until enough graded history
    exists). Also logs the signal so it can be graded later and feed the
    adaptive weight engine. Pass with_commentary=true for the LLM analyst note.
    """
    try:
        df = await get_klines(symbol, interval=interval, limit=250)
    except BinanceClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

    news = await get_news_sentiment(symbol)
    weights = adaptive_weights.get_weights(db)
    signal = await build_signal(symbol, df, news, weights=weights)

    adaptive_weights.log_signal(
        db, symbol=symbol, price=signal["price"], direction=signal["direction"],
        score=signal["score"], component_scores=signal["component_scores"],
    )

    if with_commentary:
        signal["llm_commentary"] = await analyze_with_llm(signal)

    return signal


@app.get("/api/weights")
def get_current_weights(db: Session = Depends(get_db)):
    """Shows the current adaptive weight, accuracy, and sample size for each signal component."""
    from app.database import ComponentWeight
    from app.signal_engine import DEFAULT_WEIGHTS
    rows = {cw.component: cw for cw in db.query(ComponentWeight).all()}
    out = {}
    for component, default_w in DEFAULT_WEIGHTS.items():
        cw = rows.get(component)
        out[component] = {
            "current_weight": cw.weight if cw else default_w,
            "default_weight": default_w,
            "accuracy_pct": cw.accuracy_pct if cw else None,
            "sample_size": cw.sample_size if cw else 0,
            "adapted": bool(cw and cw.sample_size >= adaptive_weights.MIN_SAMPLES_TO_ADAPT),
        }
    return out


@app.get("/api/watchlist")
def list_watchlist(db: Session = Depends(get_db)):
    items = db.query(WatchlistItem).all()
    return [{"id": w.id, "symbol": w.symbol} for w in items]


@app.post("/api/watchlist")
def add_to_watchlist(symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.upper().replace("/", "")
    existing = db.query(WatchlistItem).filter_by(symbol=symbol).first()
    if existing:
        return {"id": existing.id, "symbol": existing.symbol}
    item = WatchlistItem(symbol=symbol)
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"id": item.id, "symbol": item.symbol}


@app.delete("/api/watchlist/{item_id}")
def remove_from_watchlist(item_id: int, db: Session = Depends(get_db)):
    item = db.query(WatchlistItem).filter_by(id=item_id).first()
    if item:
        db.delete(item)
        db.commit()
    return {"deleted": True}


@app.get("/api/daily-briefs")
def get_daily_briefs(limit: int = 20, db: Session = Depends(get_db)):
    briefs = db.query(DailyBrief).order_by(DailyBrief.created_at.desc()).limit(limit).all()
    return [{
        "id": b.id, "symbol": b.symbol, "signal_score": b.signal_score,
        "direction": b.direction, "llm_commentary": b.llm_commentary,
        "traded": b.traded, "created_at": b.created_at.isoformat(),
    } for b in briefs]


@app.post("/api/run-daily-analysis-now")
async def trigger_daily_analysis():
    """Manually triggers the daily watchlist analysis job right now, instead of waiting for 08:00 UTC."""
    await daily_analysis_job()
    return {"status": "completed"}


@app.post("/api/run-maintenance-now")
async def trigger_maintenance():
    """Manually triggers grading + weight recomputation right now."""
    await maintenance_job()
    return {"status": "completed"}


@app.post("/api/paper-trade/open")
async def open_paper_trade(symbol: str, interval: str = "1h", db: Session = Depends(get_db)):
    """
    Runs analysis, then opens a paper trade automatically IF the signal
    is strong enough (|score| >= 25). Otherwise returns the signal without trading.
    This is the 'bot decides and acts' loop — but entirely on virtual money.
    """
    try:
        df = await get_klines(symbol, interval=interval, limit=250)
    except BinanceClientError as e:
        raise HTTPException(status_code=400, detail=str(e))

    news = await get_news_sentiment(symbol)
    weights = adaptive_weights.get_weights(db)
    signal = await build_signal(symbol, df, news, weights=weights)

    adaptive_weights.log_signal(
        db, symbol=symbol, price=signal["price"], direction=signal["direction"],
        score=signal["score"], component_scores=signal["component_scores"],
    )

    if signal["direction"] == "NEUTRAL / NO TRADE":
        return {"traded": False, "signal": signal}

    trade = await paper_trading.open_trade(
        db,
        symbol=symbol,
        direction=signal["direction"],
        price=signal["price"],
        signal_score=signal["score"],
        reasoning=signal["reasoning"],
    )
    return {"traded": True, "signal": signal, "trade": paper_trading._serialize(trade)}


@app.post("/api/paper-trade/close/{trade_id}")
async def close_paper_trade(trade_id: int, db: Session = Depends(get_db)):
    try:
        trade = await paper_trading.close_trade(db, trade_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return paper_trading._serialize(trade)


@app.get("/api/portfolio")
def portfolio(db: Session = Depends(get_db)):
    return paper_trading.get_portfolio(db)


# Serve the dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
