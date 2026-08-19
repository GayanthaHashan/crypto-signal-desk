"""
A fully simulated ("paper") trading account. No real money, no real
exchange order ever gets placed from this file. This is where the bot's
signals get tested safely before you'd ever consider connecting a real
Binance API key.
"""
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.database import PaperAccount, PaperTrade
from app.binance_client import get_current_price

STARTING_BALANCE = 10000.0
RISK_PER_TRADE_PCT = 0.02  # risk 2% of balance per trade (adjustable)


def get_or_create_account(db: Session, user_id: str = "default") -> PaperAccount:
    acct = db.query(PaperAccount).filter_by(user_id=user_id).first()
    if not acct:
        acct = PaperAccount(user_id=user_id, balance_usd=STARTING_BALANCE)
        db.add(acct)
        db.commit()
        db.refresh(acct)
    return acct


async def open_trade(db: Session, symbol: str, direction: str, price: float,
                      signal_score: float, reasoning: list, user_id: str = "default") -> PaperTrade:
    acct = get_or_create_account(db, user_id)
    risk_amount = acct.balance_usd * RISK_PER_TRADE_PCT
    quantity = risk_amount / price

    trade = PaperTrade(
        user_id=user_id,
        symbol=symbol,
        direction=direction,
        entry_price=price,
        quantity=quantity,
        status="OPEN",
        signal_score=signal_score,
        signal_reasoning=reasoning,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


async def close_trade(db: Session, trade_id: int, user_id: str = "default") -> PaperTrade:
    trade = db.query(PaperTrade).filter_by(id=trade_id, user_id=user_id, status="OPEN").first()
    if not trade:
        raise ValueError("Open trade not found")

    exit_price = await get_current_price(trade.symbol)
    if trade.direction == "LONG":
        pnl = (exit_price - trade.entry_price) * trade.quantity
    else:  # SHORT
        pnl = (trade.entry_price - exit_price) * trade.quantity

    trade.exit_price = exit_price
    trade.pnl_usd = pnl
    trade.status = "CLOSED"
    trade.closed_at = datetime.now(timezone.utc)

    acct = get_or_create_account(db, user_id)
    acct.balance_usd += pnl

    db.commit()
    db.refresh(trade)
    return trade


def get_portfolio(db: Session, user_id: str = "default") -> dict:
    acct = get_or_create_account(db, user_id)
    open_trades = db.query(PaperTrade).filter_by(user_id=user_id, status="OPEN").all()
    closed_trades = db.query(PaperTrade).filter_by(user_id=user_id, status="CLOSED").all()

    total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)
    wins = len([t for t in closed_trades if (t.pnl_usd or 0) > 0])
    losses = len([t for t in closed_trades if (t.pnl_usd or 0) <= 0])
    win_rate = round(wins / len(closed_trades) * 100, 1) if closed_trades else 0

    return {
        "balance_usd": round(acct.balance_usd, 2),
        "starting_balance": STARTING_BALANCE,
        "total_pnl": round(total_pnl, 2),
        "open_trades": [_serialize(t) for t in open_trades],
        "closed_trades": [_serialize(t) for t in closed_trades],
        "win_rate_pct": win_rate,
        "total_trades": len(closed_trades),
    }


def _serialize(t: PaperTrade) -> dict:
    return {
        "id": t.id,
        "symbol": t.symbol,
        "direction": t.direction,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": round(t.quantity, 6),
        "status": t.status,
        "signal_score": t.signal_score,
        "pnl_usd": round(t.pnl_usd, 2) if t.pnl_usd is not None else None,
        "opened_at": t.opened_at.isoformat() if t.opened_at else None,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
    }
