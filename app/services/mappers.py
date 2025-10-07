# app/services/mappers.py
from __future__ import annotations
from typing import Any, Dict, List
import datetime as dt

def map_funds(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    # normalize into (segment, available, utilized, etc.)
    for seg in ("equity", "derivative"):
        if seg in raw:
            d = raw[seg]
            out.append({
                "segment": seg.upper(),
                "available": d.get("available", d.get("available_cash", 0)),
                "net": d.get("net", 0),
                "updatedAt": dt.datetime.utcnow(),
            })
    return out

def map_holdings(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in raw:
        out.append({
            "tradingsymbol": h.get("tradingsymbol"),
            "instrumentToken": h.get("instrument_token"),
            "qty": h.get("quantity", 0),
            "avgPrice": h.get("average_price", 0.0),
            "lastPrice": h.get("last_price", 0.0),
            "updatedAt": dt.datetime.utcnow(),
        })
    return out

def map_positions(raw: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for bucket in ("day", "net"):
        for p in raw.get(bucket, []):
            out.append({
                "bucket": bucket,
                "tradingsymbol": p.get("tradingsymbol"),
                "instrumentToken": p.get("instrument_token"),
                "qty": p.get("quantity", 0),
                "avg": p.get("average_price", p.get("avg_price", 0.0)),
                "pnl": p.get("pnl", 0.0),
                "updatedAt": dt.datetime.utcnow(),
            })
    return out

def map_orders(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for o in raw:
        ts = o.get("order_timestamp") or o.get("timestamp")
        out.append({
            "orderId": o.get("order_id"),
            "symbol": o.get("tradingsymbol"),
            "side": o.get("transaction_type"),
            "qty": o.get("quantity", 0),
            "avgPrice": o.get("average_price", 0.0),
            "status": o.get("status"),
            "ts": ts,
        })
    return out

def map_trades(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in raw:
        ts = t.get("trade_timestamp") or t.get("timestamp")
        out.append({
            "tradeId": t.get("trade_id"),
            "orderId": t.get("order_id"),
            "symbol": t.get("tradingsymbol"),
            "qty": t.get("quantity", 0),
            "price": t.get("price", 0.0),
            "ts": ts,
        })
    return out

def map_instruments(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ins in raw:
        out.append({
            "instrumentToken": ins.get("instrument_token"),
            "tradingsymbol": ins.get("tradingsymbol"),
            "name": ins.get("name"),
            "exchange": ins.get("exchange"),
            "segment": ins.get("segment"),
            "lotSize": ins.get("lot_size"),
            "expiry": ins.get("expiry"),
            "strike": ins.get("strike"),
            "instrumentType": ins.get("instrument_type"),
        })
    return out
