"""
Pulls recent crypto headlines from free, no-API-key-required RSS feeds
(CoinDesk, Cointelegraph, Decrypt), filters to ones mentioning the coin,
and scores sentiment with VADER (lightweight, runs locally, no extra API
calls or cost).

This is a reasonable free starting point, NOT a replacement for a proper
news API. If this later feels too thin, CryptoPanic and NewsAPI both have
free tiers that just need an API key dropped into a .env file.
"""
import feedparser
import asyncio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

_analyzer = SentimentIntensityAnalyzer()

# maps common symbols to keywords used to filter relevant headlines
COIN_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance coin", "bnb"],
    "XRP": ["ripple", "xrp"],
    "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano", "ada"],
}


def _keywords_for(symbol: str) -> list[str]:
    base = symbol.upper().replace("USDT", "").replace("USD", "")
    return COIN_KEYWORDS.get(base, [base.lower()])


async def _fetch_feed(url: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, feedparser.parse, url)


async def get_news_sentiment(symbol: str, max_articles: int = 15) -> dict:
    keywords = _keywords_for(symbol)
    feeds = await asyncio.gather(*[_fetch_feed(u) for u in RSS_FEEDS], return_exceptions=True)

    matched = []
    for feed in feeds:
        if isinstance(feed, Exception) or not getattr(feed, "entries", None):
            continue
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            if any(kw in title.lower() for kw in keywords):
                matched.append({
                    "title": title,
                    "link": getattr(entry, "link", ""),
                    "published": getattr(entry, "published", ""),
                })

    matched = matched[:max_articles]

    scored = []
    total_compound = 0.0
    for article in matched:
        vs = _analyzer.polarity_scores(article["title"])
        scored.append({**article, "sentiment_score": vs["compound"]})
        total_compound += vs["compound"]

    avg_sentiment = total_compound / len(scored) if scored else 0.0

    if avg_sentiment > 0.15:
        label = "positive"
    elif avg_sentiment < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return {
        "symbol": symbol,
        "articles_found": len(scored),
        "average_sentiment": round(avg_sentiment, 4),
        "sentiment_label": label,
        "articles": scored,
    }
