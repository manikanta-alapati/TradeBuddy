# app/services/sync.py - UPDATED VERSION (Compatible with your imports)

from __future__ import annotations
import datetime as dt
from datetime import timezone, timedelta
import traceback
from collections import defaultdict

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

# Optional: lightweight "stale" check
async def _instruments_recent(db: AsyncIOMotorDatabase) -> bool:
    row = await db["meta"].find_one({"_id": "instruments_meta"})
    if not row:
        return False
    ts = row.get("updatedAt")
    if not ts:
        return False
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
    skip_embeddings: bool = True,
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
        all_orders = kite.get_orders()
    
        # Build trade history from completed orders
        trade_history = []
        for order in all_orders:
            if order.get('status') == 'COMPLETE' and order.get('filled_quantity', 0) > 0:
                trade_history.append({
                    'tradeId': order.get('order_id'),
                    'orderId': order.get('order_id'),
                    'symbol': order.get('tradingsymbol'),
                    'side': order.get('transaction_type'),
                    'qty': order.get('filled_quantity'),
                    'price': order.get('average_price'),
                    'ts': order.get('order_timestamp')
                })
    
        # Upsert trades
        await upsert_trades(db, user_id, trade_history)
        changed.append({"doc": "trades", "count": len(trade_history)})
    
        # Also save all orders
        await upsert_orders(db, user_id, map_orders(all_orders))
        changed.append({"doc": "orders", "count": len(all_orders)})
    
    except Exception as e:
        changed.append({"doc": "orders", "error": str(e)})

    return {
        "userId": str(user_id),
        "syncedAt": dt.datetime.now(timezone.utc).isoformat(),
        "updated": changed,
        "embeddings": "skipped" if skip_embeddings else "handled elsewhere"
    }


async def build_embeddings_for_user(db: AsyncIOMotorDatabase, user_id: ObjectId) -> Dict[str, Any]:
    """
    Complete embeddings generation pipeline with POSITIONS + TRADE HISTORY.
    
    UPDATED: Now includes positions (active trades) in portfolio summary.
    """
    # ✅ USE YOUR ACTUAL IMPORTS (not the wrong ones I gave you!)
    from app.services.llm import embed_text
    from app.services.vector import upsert_embedding
    from datetime import datetime, timezone
    
    embeddings_created = 0
    errors = []
    
    try:
        # ============================================
        # 1. PORTFOLIO SUMMARY (Holdings + Positions)
        # ============================================
        
        # Get holdings (long-term)
        holdings = await db["holdings"].find({"userId": user_id}).to_list(None)
        
        # Get positions (active trades)
        positions = await db["positions"].find({"userId": user_id}).to_list(None)
        
        if holdings or positions:
            # Calculate holdings metrics
            holdings_value = 0
            holdings_investment = 0
            holdings_text_parts = []
            
            for h in holdings:
                qty = h.get("qty", 0)
                avg_price = h.get("avgPrice", 0)
                last_price = h.get("lastPrice", 0)
                symbol = h.get("tradingsymbol", "UNKNOWN")
                
                investment = qty * avg_price
                current_value = qty * last_price
                pnl = current_value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                
                holdings_investment += investment
                holdings_value += current_value
                
                holdings_text_parts.append(
                    f"HOLDING - {symbol}: {qty} shares @ avg Rs.{avg_price:.2f}, "
                    f"current Rs.{last_price:.2f}, "
                    f"P&L: Rs.{pnl:,.2f} ({pnl_pct:+.2f}%)"
                )
            
            # Calculate positions metrics
            positions_value = 0
            positions_investment = 0
            positions_text_parts = []
            
            for p in positions:
                qty = p.get("quantity", 0)
                avg_price = p.get("average_price", 0)
                last_price = p.get("last_price", 0)
                symbol = p.get("tradingsymbol", "UNKNOWN")
                product = p.get("product", "")
                
                investment = qty * avg_price
                current_value = qty * last_price
                pnl = current_value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                
                positions_investment += investment
                positions_value += current_value
                
                positions_text_parts.append(
                    f"POSITION ({product}) - {symbol}: {qty} @ avg Rs.{avg_price:.2f}, "
                    f"current Rs.{last_price:.2f}, "
                    f"P&L: Rs.{pnl:,.2f} ({pnl_pct:+.2f}%)"
                )
            
            # Total calculations
            total_investment = holdings_investment + positions_investment
            total_value = holdings_value + positions_value
            total_pnl = total_value - total_investment
            total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
            
            # Build comprehensive summary
            portfolio_summary = (
                "COMPLETE PORTFOLIO OVERVIEW:\n"
                "\n"
                "HOLDINGS (Long-term investments):\n"
                f"Count: {len(holdings)} stocks\n"
                f"Investment: Rs.{holdings_investment:,.2f}\n"
                f"Current Value: Rs.{holdings_value:,.2f}\n"
                f"P&L: Rs.{holdings_value - holdings_investment:,.2f}\n"
                "\n"
            )
            
            if holdings_text_parts:
                portfolio_summary += "Holdings Detail:\n" + "\n".join(holdings_text_parts) + "\n\n"
            
            portfolio_summary += (
                "POSITIONS (Active trades):\n"
                f"Count: {len(positions)} trades\n"
                f"Investment: Rs.{positions_investment:,.2f}\n"
                f"Current Value: Rs.{positions_value:,.2f}\n"
                f"P&L: Rs.{positions_value - positions_investment:,.2f}\n"
                "\n"
            )
            
            if positions_text_parts:
                portfolio_summary += "Positions Detail:\n" + "\n".join(positions_text_parts) + "\n\n"
            
            portfolio_summary += (
                "TOTAL PORTFOLIO:\n"
                f"Total Investment: Rs.{total_investment:,.2f}\n"
                f"Total Current Value: Rs.{total_value:,.2f}\n"
                f"Total P&L: Rs.{total_pnl:,.2f} ({total_pnl_pct:+.2f}%)\n"
                "\n"
                f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            
            # Generate embedding
            portfolio_vector = embed_text(portfolio_summary)

            await upsert_embedding(
                db,
                user_id=user_id,
                kind="portfolio_summary",
                doc_id=f"portfolio-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                vector=portfolio_vector,
                chunk=portfolio_summary,
                metadata={
                    "totalValue": total_value,
                    "totalPnL": total_pnl,
                    "holdingsCount": len(holdings),
                    "positionsCount": len(positions),
                    "generatedAt": datetime.now(timezone.utc).isoformat()
                }
            )
            
            embeddings_created += 1
            print(f"✅ Created portfolio summary embedding")
        
        # ============================================
        # 2. TRADE HISTORY EMBEDDINGS
        # ============================================
        
        six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
        
        trades = await db["trades"].find({
            "userId": user_id,
            "ts": {"$gte": six_months_ago}
        }).sort("ts", -1).to_list(None)
        
        if trades:
            print(f"📊 Found {len(trades)} trades in last 6 months")
            
            trades_by_month = defaultdict(list)
            
            for trade in trades:
                ts = trade.get("ts")
                if ts:
                    month_key = ts.strftime("%Y-%m")
                    trades_by_month[month_key].append(trade)
            
            for month, month_trades in trades_by_month.items():
                buy_trades = []
                sell_trades = []
                
                for t in month_trades:
                    side = str(t.get("side", "")).upper()
                    if "BUY" in side:
                        buy_trades.append(t)
                    elif "SELL" in side:
                        sell_trades.append(t)
                
                buy_value = sum(t.get("qty", 0) * t.get("price", 0) for t in buy_trades)
                sell_value = sum(t.get("qty", 0) * t.get("price", 0) for t in sell_trades)
                
                trade_summary_lines = [
                    f"TRADE ACTIVITY FOR {month}:",
                    f"Total Trades: {len(month_trades)}",
                    f"Buys: {len(buy_trades)} trades worth Rs.{buy_value:,.0f}",
                    f"Sells: {len(sell_trades)} trades worth Rs.{sell_value:,.0f}",
                    f"Net Cash Flow: Rs.{sell_value - buy_value:,.0f}",
                    ""
                ]
                
                if buy_trades:
                    trade_summary_lines.append("PURCHASES:")
                    for t in buy_trades:
                        sym = t.get("symbol", "UNKNOWN")
                        qty = t.get("qty", 0)
                        price = t.get("price", 0)
                        ts = t.get("ts")
                        date_str = ts.strftime("%Y-%m-%d") if ts else "Unknown"
                        trade_summary_lines.append(
                            f"  {date_str}: Bought {qty} shares of {sym} @ Rs.{price:.2f}"
                        )
                
                if sell_trades:
                    trade_summary_lines.append("\nSALES:")
                    for t in sell_trades:
                        sym = t.get("symbol", "UNKNOWN")
                        qty = t.get("qty", 0)
                        price = t.get("price", 0)
                        ts = t.get("ts")
                        date_str = ts.strftime("%Y-%m-%d") if ts else "Unknown"
                        trade_summary_lines.append(
                            f"  {date_str}: Sold {qty} shares of {sym} @ Rs.{price:.2f}"
                        )
                
                trade_summary = "\n".join(trade_summary_lines)
                
                trade_vector = embed_text(trade_summary)
                
                await upsert_embedding(
                    db,
                    user_id=user_id,
                    kind="trade_history",
                    doc_id=f"trades-{month}",
                    vector=trade_vector,
                    chunk=trade_summary,
                    metadata={
                        "month": month,
                        "tradesCount": len(month_trades),
                        "buyCount": len(buy_trades),
                        "sellCount": len(sell_trades),
                        "generatedAt": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                embeddings_created += 1
                print(f"✅ Created trade history embedding for {month}")
        else:
            print("⚠️ No trades found in last 6 months")
        
        # ============================================
        # 3. FUNDS SUMMARY
        # ============================================
        
        funds = await db["funds"].find({"userId": user_id}).to_list(None)
        
        if funds:
            funds_parts = []
            for f in funds:
                segment = f.get("segment", "EQUITY")
                available = f.get("available", 0)
                if isinstance(available, dict):
                    available = available.get("cash", 0)
                net = f.get("net", 0)
                
                funds_parts.append(f"{segment}: Available Rs.{available:,.2f}, Net Rs.{net:,.2f}")
            
            funds_summary = "ACCOUNT FUNDS:\n" + "\n".join(funds_parts)
            funds_vector = embed_text(funds_summary)

            await upsert_embedding(
                db,
                user_id=user_id,
                kind="funds_summary",
                doc_id=f"funds-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                vector=funds_vector,
                chunk=funds_summary,
                metadata={"generatedAt": datetime.now(timezone.utc).isoformat()}
            )
            
            embeddings_created += 1
            print(f"✅ Created funds summary embedding")
        
    except Exception as e:
        error_detail = traceback.format_exc()
        errors.append(str(e))
        print(f"[Embeddings] ERROR for user {user_id}:")
        print(error_detail)
    
    return {
        "userId": str(user_id),
        "embeddingsCreated": embeddings_created,
        "errors": errors,
        "status": "success" if embeddings_created > 0 else "no_data"
    }


async def list_active_users(db: AsyncIOMotorDatabase) -> List[ObjectId]:
    """Get list of all users with active Zerodha connections."""
    cursor = db["connections"].find(
        {"provider": "zerodha", "enabled": True},
        {"userId": 1}
    )
    
    user_ids = []
    async for doc in cursor:
        user_ids.append(doc["userId"])
    
    return user_ids