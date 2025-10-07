# app/services/summarize.py
from typing import Dict, Any
from openai import OpenAI
from app.settings import settings

client = None
def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=settings.openai_api_key)
    return client

MODEL = "gpt-4o-mini"

SYSTEM = """You write terse, factual portfolio summaries. 
No fluff. Use INR currency symbol ₹ when appropriate. 
If data seems incomplete, acknowledge it plainly."""

def summarize_symbol_month(symbol: str, month: str, facts: Dict[str, Any]) -> str:
    """
    facts: {
      "gross_pnl": float, "net_pnl": float, "trades_count": int,
      "win_rate": float, "avg_price": float, "notes": str|None, ...
    }
    """
    prompt = f"""Summarize performance for {symbol} in {month}.
Facts (JSON):
{facts}

Rules:
- One short paragraph (2–4 sentences).
- Start with net P&L with sign, then key drivers (e.g., win rate, notable trades).
- If notes exist, include briefly.
- If data is missing, say 'data incomplete'."""
    cli = get_client()
    resp = cli.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()
