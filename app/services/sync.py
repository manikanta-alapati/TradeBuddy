# app/services/sync.py
from __future__ import annotations
import datetime as dt
from datetime import timezone
from typing import Dict, Any, List
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.kite_client import build_kite_client
from app.services.mappers import (
    map_instruments, map_funds, map_holdings, map_positions, map_orders, map_trades
)
from app.services.upserts import (
    upsert_instruments,
    upsert_funds,
    upsert_holdings,
    upsert_positions,
    upsert_orders,
    upsert_trades,
)

# Optional: lightweight “stale” check (can be improved later)
async def _instruments_recent(db: AsyncIOMotorDatabase) -> bool:
    row = await db["meta"].find_one({"_id": "instruments_meta"})
    if not row:
        return False
    ts = row.get("updatedAt")
    if not ts:
        return False
    # consider fresh if updated within the last 3 days
    return (dt.datetime.now(timezone.utc) - ts) < dt.timedelta(days=3)

async def _mark_instruments_updated(db: AsyncIOMotorDatabase):
    await db["meta"].update_one(
        {"_id": "instruments_meta"},
        {"$set": {"updatedAt": dt.datetime.now(timezone.utc)}},
        upsert=True
    )

async def run_incremental_sync(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    *,
    force_instruments: bool = False,
    skip_embeddings: bool = True,   # default fast
) -> Dict[str, Any]:
    """
    Fast path sync:
    - Optionally pulls master instruments (only when forced).
    - Always pulls funds/holdings/positions/orders/trades and upserts.
    - Embeddings are intentionally skipped here; scheduler can enqueue background job.
    """
    changed: List[Dict[str, Any]] = []

    # 0) connection
    conn = await db["connections"].find_one({"userId": user_id, "provider": "zerodha", "enabled": True})
    if not conn:
        return {"userId": str(user_id), "syncedAt": dt.datetime.now(timezone.utc).isoformat(), "updated": changed, "note": "no-connection"}

    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))

    # 1) instruments (optional & heavy)
    if force_instruments:
        try:
            if not await _instruments_recent(db):
                instruments = map_instruments(kite.get_instruments())
                await upsert_instruments(db, instruments)
                await _mark_instruments_updated(db)
                changed.append({"doc": "instruments"})
        except Exception as e:
            # Don't fail the whole sync if instruments are slow
            changed.append({"doc": "instruments", "error": str(e)})

    # 2) core user data (fast)
    try:
        await upsert_funds(db, user_id, map_funds(kite.get_funds()))
        changed.append({"doc": "funds"})
    except Exception as e:
        changed.append({"doc": "funds", "error": str(e)})

    try:
        await upsert_holdings(db, user_id, map_holdings(kite.get_holdings()))
        changed.append({"doc": "holdings"})
    except Exception as e:
        changed.append({"doc": "holdings", "error": str(e)})

    try:
        await upsert_positions(db, user_id, map_positions(kite.get_positions()))
        changed.append({"doc": "positions"})
    except Exception as e:
        changed.append({"doc": "positions", "error": str(e)})

    try:
        await upsert_orders(db, user_id, map_orders(kite.get_orders()))
        changed.append({"doc": "orders"})
    except Exception as e:
        changed.append({"doc": "orders", "error": str(e)})

    try:
        await upsert_trades(db, user_id, map_trades(kite.get_trades()))
        changed.append({"doc": "trades"})
    except Exception as e:
        changed.append({"doc": "trades", "error": str(e)})

    # 3) embeddings are skipped here; background job handles it
    return {
        "userId": str(user_id),
        "syncedAt": dt.datetime.now(timezone.utc).isoformat(),
        "updated": changed,
        "embeddings": "skipped" if skip_embeddings else "handled elsewhere"
    }

async def build_embeddings_for_user(db: AsyncIOMotorDatabase, user_id: ObjectId) -> Dict[str, Any]:
    """
    Minimal placeholder. You can implement:
      - compute_symbol_month_rollups(...)
      - summarize_symbol_month(...)
      - embed_text(...)
      - upsert_embedding(...)
    For now we return a no-op result so your scheduler/job path is stable.
    """
    # TODO: implement your real summarization + embedding pipeline here.
    return {"userId": str(user_id), "built": 0, "status": "noop"}