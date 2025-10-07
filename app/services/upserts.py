# app/services/upserts.py
from __future__ import annotations
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
import datetime as dt
from decimal import Decimal
from pymongo import UpdateOne

# Optional: support numpy scalars if they appear anywhere
try:
    import numpy as np
    _NUMPY = True
except Exception:
    _NUMPY = False


def _to_mongo_safe(value: Any) -> Any:
    """
    Recursively convert values so MongoDB can encode them.
    - date -> datetime (UTC midnight)
    - tz-aware datetime -> naive UTC datetime
    - Decimal -> float
    - numpy scalars -> python scalars
    - dict/list -> recurse
    """
    # 1) datetimes / dates
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            # convert to UTC & drop tzinfo
            value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value

    if isinstance(value, dt.date):
        # convert date to UTC midnight
        return dt.datetime.combine(value, dt.time.min)

    # 2) Decimal
    if isinstance(value, Decimal):
        return float(value)

    # 3) numpy scalars
    if _NUMPY:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)

    # 4) containers
    if isinstance(value, dict):
        return {k: _to_mongo_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_mongo_safe(v) for v in value]

    return value


def _normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy then recursively sanitize for Mongo."""
    return _to_mongo_safe(dict(doc))


def normalize_instrument(instr: dict) -> dict:
    d = instr.copy()
    if isinstance(d.get("expiry"), dt.date) and not isinstance(d["expiry"], dt.datetime):
        d["expiry"] = dt.datetime.combine(d["expiry"], dt.time.min)
    return d

# ---------- UPSERT HELPERS ----------

async def upsert_instruments(db, instruments):
    col = db["instruments"]
    ops = []
    for d in instruments:
        d = normalize_instrument(d)
        key = {"instrumentToken": d["instrumentToken"]}
        ops.append(UpdateOne(key, {"$set": d}, upsert=True))
        if len(ops) == 1000:
            await col.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        await col.bulk_write(ops, ordered=False)

async def upsert_funds(db: AsyncIOMotorDatabase, user_id: ObjectId, docs: List[Dict[str, Any]]):
    col = db["funds"]
    for d in docs:
        d2 = _normalize_doc(d)
        d2["userId"] = user_id
        # Segment generally identifies a single row per user/segment
        await col.update_one({"userId": user_id, "segment": d2.get("segment", "EQ")},
                             {"$set": d2}, upsert=True)


async def upsert_holdings(db: AsyncIOMotorDatabase, user_id: ObjectId, docs: List[Dict[str, Any]]):
    col = db["holdings"]
    for d in docs:
        d2 = _normalize_doc(d)
        d2["userId"] = user_id
        key = {"userId": user_id, "tradingsymbol": d2.get("tradingsymbol")}
        await col.update_one(key, {"$set": d2}, upsert=True)


async def upsert_positions(db: AsyncIOMotorDatabase, user_id: ObjectId, docs: List[Dict[str, Any]]):
    """
    Zerodha positions API typically returns two buckets: day / net.
    Your mapper should set d['bucket'] to 'day' or 'net'.
    """
    col = db["positions"]
    for d in docs:
        d2 = _normalize_doc(d)
        d2["userId"] = user_id
        key = {
            "userId": user_id,
            "tradingsymbol": d2.get("tradingsymbol"),
            "bucket": d2.get("bucket")  # 'day' or 'net'
        }
        await col.update_one(key, {"$set": d2}, upsert=True)


async def upsert_orders(db: AsyncIOMotorDatabase, user_id: ObjectId, docs: List[Dict[str, Any]]):
    """
    Orders can include multiple datetime-ish fields depending on status:
    - order_timestamp, exchange_timestamp, placed_by, etc.
    The normalizer will make them Mongo-safe automatically.
    """
    col = db["orders"]
    for d in docs:
        d2 = _normalize_doc(d)
        d2["userId"] = user_id
        key = {"orderId": d2.get("orderId")}
        await col.update_one(key, {"$set": d2}, upsert=True)


async def upsert_trades(db: AsyncIOMotorDatabase, user_id: ObjectId, docs: List[Dict[str, Any]]):
    """
    Trades typically have trade_timestamp; sometimes Zerodha returns date-like objects.
    The normalizer covers those.
    """
    col = db["trades"]
    for d in docs:
        d2 = _normalize_doc(d)
        d2["userId"] = user_id
        key = {"tradeId": d2.get("tradeId")}
        await col.update_one(key, {"$set": d2}, upsert=True)
