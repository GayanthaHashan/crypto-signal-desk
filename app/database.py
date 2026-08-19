from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./paper_trading.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PaperAccount(Base):
    __tablename__ = "paper_accounts"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, default="default", unique=True)
    balance_usd = Column(Float, default=10000.0)  # starting virtual balance
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PaperTrade(Base):
    __tablename__ = "paper_trades"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, default="default")
    symbol = Column(String)
    direction = Column(String)          # LONG or SHORT
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float)
    status = Column(String, default="OPEN")   # OPEN or CLOSED
    signal_score = Column(Float, nullable=True)
    signal_reasoning = Column(JSON, nullable=True)
    opened_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at = Column(DateTime, nullable=True)
    pnl_usd = Column(Float, nullable=True)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, default="default")
    symbol = Column(String)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SignalRecord(Base):
    """
    Every signal the engine produces gets logged here with its raw
    per-component scores. ~24h later, the grading job checks what price
    actually did and marks each component as having been 'right' or
    'wrong' about direction. That history is what the adaptive weight
    engine learns from — this table IS the bot's track record.
    """
    __tablename__ = "signal_records"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    price_at_signal = Column(Float)
    direction = Column(String)
    final_score = Column(Float)
    component_scores = Column(JSON)   # {"rsi": 0.4, "macd": -0.2, ...} raw -1..1 scores
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    graded = Column(String, default="PENDING")   # PENDING, GRADED
    outcome_price = Column(Float, nullable=True)
    actual_direction = Column(String, nullable=True)   # UP, DOWN, FLAT
    graded_at = Column(DateTime, nullable=True)


class ComponentWeight(Base):
    """Current adaptive weight for each signal component, learned from SignalRecord history."""
    __tablename__ = "component_weights"
    id = Column(Integer, primary_key=True)
    component = Column(String, unique=True)
    weight = Column(Float)
    accuracy_pct = Column(Float, nullable=True)
    sample_size = Column(Integer, default=0)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DailyBrief(Base):
    """Stored output of the daily scheduled analysis + LLM commentary for a watchlist symbol."""
    __tablename__ = "daily_briefs"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    signal_score = Column(Float)
    direction = Column(String)
    llm_commentary = Column(String)
    traded = Column(String, default="NO")   # YES / NO
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
