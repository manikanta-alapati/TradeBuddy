# app/services/rollup.py
from typing import Dict, Any, List, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

def ym(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

async def compute_symbol_month_rollups(
    db: AsyncIOMotorDatabase, user_id: ObjectId, month: str | None = None
) -> Dict[str, Dict[str, Any]]:
    """
    Returns { "SYMBOL-YYYY-MM": {facts...}, ... }
    Pulls from trades collection if available; otherwise returns an empty dict.
    """
    match: Dict[str, Any] = {"userId": user_id}
    if month:
        y, m = month.split("-")
        match["ts"] = {
            "$gte": datetime(int(y), int(m), 1),
            "$lt": datetime(int(y) + (1 if m == "12" else 0), (int(m) % 12) + 1, 1),
        }

    pipeline = [
        {"$match": match},
        {"$project": {"symbol": 1, "price": 1, "qty": 1, "ts": 1, "amount": {"$multiply": ["$price", "$qty"]}}},
        {
            "$group": {
                "_id": {"symbol": "$symbol", "ym": {"$dateToString": {"format": "%Y-%m", "date": "$ts"}}},
                "trades_count": {"$sum": 1},
                "gross_amount": {"$sum": "$amount"},
                "avg_price": {"$avg": "$price"},
            }
        },
    ]
    out: Dict[str, Dict[str, Any]] = {}
    async for row in db["trades"].aggregate(pipeline):
        symbol = row["_id"]["symbol"]
        key = f"{symbol}-{row['_id']['ym']}"
        out[key] = {
            "symbol": symbol,
            "trades_count": row["trades_count"],
            "gross_pnl": row["gross_amount"],   # placeholder; real pnl requires buy/sell netting + fees
            "net_pnl": row["gross_amount"],     # placeholder
            "avg_price": row.get("avg_price"),
            "win_rate": None,                    # compute once you have side & realized pnl per trade
            "notes": None,
        }
    return out
