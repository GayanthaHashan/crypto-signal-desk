"""
Calls an LLM to add a written, analyst-style read on top of the
quantitative signal — reasoning through it the way a technical analyst
would, using named strategy frameworks (trend-following, mean-reversion,
breakout, support/resistance confluence). It does NOT get separate market
data of its own; it only reasons over the numbers signal_engine already
computed, so it can't hallucinate prices or news that weren't actually
found.

Supports two providers, tried in this order:
1. GEMINI_API_KEY — Google's Gemini API, which has a genuine free tier
   (no credit card needed). This is the default for anyone not ready to
   pay for API usage. Get a key at https://aistudio.google.com/apikey
2. ANTHROPIC_API_KEY — paid, but higher quality if you want to switch later.

If neither key is set, this layer is skipped and you just get the
quantitative signal without commentary — everything else still works.
"""
import os

GEMINI_MODEL = "gemini-3.6-flash"
ANTHROPIC_MODEL = "claude-sonnet-5"


def _build_prompt(signal: dict) -> str:
    return f"""You are a technical analyst reviewing an automated trading signal.
You are NOT placing any trade — you're writing a short, honest daily note for a trader
who will read this alongside the raw signal below.

Symbol: {signal['symbol']}
Current price: {signal['price']}
Composite score: {signal['score']} (-100 bearish to +100 bullish)
Direction: {signal['direction']}
Confidence: {signal['confidence']}%

Component breakdown:
{chr(10).join(signal['reasoning'])}

Fibonacci: nearest level {signal['fibonacci']['nearest_level']} at {signal['fibonacci']['nearest_level_price']:.2f}
Candlestick patterns detected: {[p['pattern'] for p in signal['candlestick_patterns']] or 'none'}
News sentiment: {signal['news_summary']['label']} ({signal['news_summary']['articles_found']} articles)

Write a 3-5 sentence analyst note that:
1. States which classic strategy framework(s) this setup resembles (trend-following, mean-reversion,
   breakout, support/resistance confluence, or "no clean setup") and why, referencing the actual numbers above.
2. Flags the single biggest risk to this read (e.g. conflicting indicators, thin news sample, choppy conditions).
3. Does NOT tell the user to definitely take the trade — end with what would invalidate this view.

Be direct and specific. No generic disclaimers beyond what's asked for. No markdown headers."""


async def _try_gemini(prompt: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    import httpx
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
    if resp.status_code != 200:
        return f"(Gemini API error {resp.status_code}: {resp.text[:200]})"
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return None


async def _try_anthropic(prompt: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def analyze_with_llm(signal: dict) -> str | None:
    prompt = _build_prompt(signal)

    result = await _try_gemini(prompt)
    if result is not None:
        return result

    result = await _try_anthropic(prompt)
    if result is not None:
        return result

    return None
