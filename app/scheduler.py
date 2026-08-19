"""
Runs two jobs on a schedule using APScheduler (in-process — fine for a
single free-tier instance; if you ever scale to multiple instances, move
this to a proper task queue so jobs don't run multiple times):

1. daily_analysis_job — once a day, analyzes every symbol on the watchlist,
   logs the signal for future grading, gets an LLM commentary note, stores
   it as a DailyBrief, and opens a paper trade if the signal is strong.
2. maintenance_job — grades signals that are now old enough to check, then
   recomputes each component's adaptive weight from the updated track record.
   Runs more frequently so weights stay current.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timezone

from app.database import SessionLocal, WatchlistItem, DailyBrief
from app.binance_client import get_klines, BinanceClientError
from app.news_sentiment import get_news_sentiment
from app.signal_engine import build_signal
from app.llm_analyst import analyze_with_llm
from app import adaptive_weights, paper_trading

scheduler = AsyncIOScheduler()


async def daily_analysis_job():
    db = SessionLocal()
    try:
        symbols = [w.symbol for w in db.query(WatchlistItem).all()]
        weights = adaptive_weights.get_weights(db)

        for symbol in symbols:
            try:
                df = await get_klines(symbol, interval="1h", limit=250)
                news = await get_news_sentiment(symbol)
                signal = await build_signal(symbol, df, news, weights=weights)

                adaptive_weights.log_signal(
                    db, symbol=symbol, price=signal["price"],
                    direction=signal["direction"], score=signal["score"],
                    component_scores=signal["component_scores"],
                )

                commentary = await analyze_with_llm(signal)

                traded = "NO"
                if signal["direction"] != "NEUTRAL / NO TRADE":
                    await paper_trading.open_trade(
                        db, symbol=symbol, direction=signal["direction"],
                        price=signal["price"], signal_score=signal["score"],
                        reasoning=signal["reasoning"],
                    )
                    traded = "YES"

                db.add(DailyBrief(
                    symbol=symbol, signal_score=signal["score"],
                    direction=signal["direction"],
                    llm_commentary=commentary or "(no LLM commentary - ANTHROPIC_API_KEY not set)",
                    traded=traded,
                ))
                db.commit()
            except BinanceClientError:
                continue
    finally:
        db.close()


async def maintenance_job():
    db = SessionLocal()
    try:
        await adaptive_weights.grade_pending_signals(db)
        adaptive_weights.recompute_weights(db)
    finally:
        db.close()


def start_scheduler():
    # daily analysis at 08:00 UTC; maintenance every 6 hours
    scheduler.add_job(daily_analysis_job, "cron", hour=8, minute=0, id="daily_analysis")
    scheduler.add_job(maintenance_job, "interval", hours=6, id="maintenance")
    scheduler.start()
