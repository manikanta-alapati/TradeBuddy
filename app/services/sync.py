# app/services/sync.py
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

# Optional: lightweight "stale" check (can be improved later)
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
        all_orders = kite.get_orders()
    
    # Build trade history from completed orders
        trade_history = []
        for order in all_orders:
        # Only include COMPLETE orders as trades
            if order.get('status') == 'COMPLETE' and order.get('filled_quantity', 0) > 0:
                trade_history.append({
                    'tradeId': order.get('order_id'),
                    'orderId': order.get('order_id'),
                    'symbol': order.get('tradingsymbol'),
                    'side': order.get('transaction_type'),  # BUY or SELL
                    'qty': order.get('filled_quantity'),
                    'price': order.get('average_price'),
                    'ts': order.get('order_timestamp')
            })
    
    # Upsert trades (from orders data)
        await upsert_trades(db, user_id, trade_history)
        changed.append({"doc": "trades", "count": len(trade_history)})
    
    # Also save all orders separately
        await upsert_orders(db, user_id, map_orders(all_orders))
        changed.append({"doc": "orders", "count": len(all_orders)})
    
    except Exception as e:
        changed.append({"doc": "orders", "error": str(e)})
        
        
        

    # 3) embeddings are skipped here; background job handles it
    return {
        "userId": str(user_id),
        "syncedAt": dt.datetime.now(timezone.utc).isoformat(),
        "updated": changed,
        "embeddings": "skipped" if skip_embeddings else "handled elsewhere"
    }

async def build_embeddings_for_user(db: AsyncIOMotorDatabase, user_id: ObjectId) -> Dict[str, Any]:
    """
    Complete embeddings generation pipeline with TRADE HISTORY:
    1. Generate portfolio summary
    2. Generate trade history summaries (by month)
    3. Generate symbol-month summaries
    4. Convert to embeddings
    5. Store in MongoDB
    
    This makes your portfolio data AND trade history searchable with semantic queries.
    """
    from app.services.llm import embed_text
    from app.services.vector import upsert_embedding
    from app.services.rollup import compute_symbol_month_rollups
    from app.services.summarize import summarize_symbol_month
    from datetime import datetime, timezone
    
    embeddings_created = 0
    errors = []
    
    # Get user phone for embedding metadata
    user = await db["users"].find_one({"_id": user_id})
    phone = user.get("phone") if user else None
    
    try:
        # ============================================
        # 1. PORTFOLIO SUMMARY EMBEDDING
        # ============================================
        
        # Fetch current holdings
        holdings = await db["holdings"].find({"userId": user_id}).to_list(None)
        
        if holdings:
            # Calculate portfolio metrics
            total_value = 0
            total_investment = 0
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
                allocation = (current_value / total_value * 100) if total_value > 0 else 0
                
                total_investment += investment
                total_value += current_value
                
                # Create human-readable text for this holding
                holdings_text_parts.append(
                    f"{symbol}: {qty} shares @ avg ₹{avg_price:.2f}, "
                    f"current ₹{last_price:.2f}, "
                    f"P&L: ₹{pnl:,.2f} ({pnl_pct:+.2f}%), "
                    f"Allocation: {allocation:.1f}% of portfolio"
                )
            
            # Recalculate allocations now that we have total_value
            for i, h in enumerate(holdings):
                qty = h.get("qty", 0)
                last_price = h.get("lastPrice", 0)
                current_value = qty * last_price
                allocation = (current_value / total_value * 100) if total_value > 0 else 0
                
                # Update allocation in text
                symbol = h.get("tradingsymbol", "UNKNOWN")
                avg_price = h.get("avgPrice", 0)
                investment = qty * avg_price
                pnl = current_value - investment
                pnl_pct = (pnl / investment * 100) if investment > 0 else 0
                
                holdings_text_parts[i] = (
                    f"{symbol}: {qty} shares @ avg ₹{avg_price:.2f}, "
                    f"current ₹{last_price:.2f}, "
                    f"P&L: ₹{pnl:,.2f} ({pnl_pct:+.2f}%), "
                    f"Allocation: {allocation:.1f}%"
                )
            
            # Overall portfolio summary
            total_pnl = total_value - total_investment
            total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
            holdings_breakdown = "\n".join(holdings_text_parts)

            portfolio_summary = (
                "CURRENT PORTFOLIO OVERVIEW:\n"
                f"Total Holdings: {len(holdings)} stocks\n"
                f"Total Investment: ₹{total_investment:,.2f}\n"
                f"Current Value: ₹{total_value:,.2f}\n"
                f"Total P&L: ₹{total_pnl:,.2f} ({total_pnl_pct:+.2f}%)\n"
                "\n"
                "DETAILED HOLDINGS:\n"
                f"{holdings_breakdown}\n"
                "\n"
                f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            
            # Generate embedding and store
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
                    "generatedAt": datetime.now(timezone.utc).isoformat()
                }
            )
            
            embeddings_created += 1
            print(f"✅ Created portfolio summary embedding")
        
        # ============================================
        # 2. TRADE HISTORY EMBEDDINGS (NEW!)
        # ============================================
        
        # Get last 6 months of trades
        six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
        
        trades = await db["trades"].find({
            "userId": user_id,
            "ts": {"$gte": six_months_ago}
        }).sort("ts", -1).to_list(None)
        
        if trades:
            print(f"📊 Found {len(trades)} trades in last 6 months")
            
            # Group trades by month
            trades_by_month = defaultdict(list)
            
            for trade in trades:
                ts = trade.get("ts")
                if ts:
                    month_key = ts.strftime("%Y-%m")  # "2024-09"
                    trades_by_month[month_key].append(trade)
            
            # Create embeddings for each month
            for month, month_trades in trades_by_month.items():
                # Separate buys and sells
                buy_trades = []
                sell_trades = []
                
                for t in month_trades:
                    # Check side field (might be in different formats)
                    side = str(t.get("side", "")).upper()
                    if "BUY" in side:
                        buy_trades.append(t)
                    elif "SELL" in side:
                        sell_trades.append(t)
                
                # Calculate totals
                buy_value = sum(t.get("qty", 0) * t.get("price", 0) for t in buy_trades)
                sell_value = sum(t.get("qty", 0) * t.get("price", 0) for t in sell_trades)
                
                # Build detailed summary
                trade_summary_lines = [
                    f"TRADE ACTIVITY FOR {month}:",
                    f"Total Trades: {len(month_trades)}",
                    f"Buys: {len(buy_trades)} trades worth ₹{buy_value:,.0f}",
                    f"Sells: {len(sell_trades)} trades worth ₹{sell_value:,.0f}",
                    f"Net Cash Flow: ₹{sell_value - buy_value:,.0f}",
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
                            f"  {date_str}: Bought {qty} shares of {sym} @ ₹{price:.2f} = ₹{qty*price:,.0f}"
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
                            f"  {date_str}: Sold {qty} shares of {sym} @ ₹{price:.2f} = ₹{qty*price:,.0f}"
                        )
                
                trade_summary_lines.append(f"\n[Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
                trade_summary = "\n".join(trade_summary_lines)
                
                # Generate embedding
                trade_vector = embed_text(trade_summary)
                
                await upsert_embedding(
                    db,
                    user_id=user_id,
                    kind="trade_history",  # New kind for trade history!
                    doc_id=f"trades-{month}",
                    vector=trade_vector,
                    chunk=trade_summary,
                    metadata={
                        "month": month,
                        "tradesCount": len(month_trades),
                        "buyCount": len(buy_trades),
                        "sellCount": len(sell_trades),
                        "buyValue": buy_value,
                        "sellValue": sell_value,
                        "generatedAt": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                embeddings_created += 1
                print(f"✅ Created trade history embedding for {month}: {len(month_trades)} trades")
        else:
            print("⚠️ No trades found in last 6 months")
        
        # ============================================
        # 3. FUNDS SUMMARY EMBEDDING
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
                
                funds_parts.append(
                    f"{segment}: Available ₹{available:,.2f}, Net ₹{net:,.2f}"
                )
            
            funds_breakdown = "\n".join(funds_parts)

            funds_summary = (
                "ACCOUNT FUNDS:\n"
                f"{funds_breakdown}\n"
                "\n"
                f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            )
            
            funds_vector = embed_text(funds_summary)

            await upsert_embedding(
                db,
                user_id=user_id,
                kind="funds_summary",
                doc_id=f"funds-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                vector=funds_vector,
                chunk=funds_summary,
                metadata={
                    "generatedAt": datetime.now(timezone.utc).isoformat()
                }
            )
            
            embeddings_created += 1
            print(f"✅ Created funds summary embedding")
        
        # ============================================
        # 4. POSITIONS SUMMARY (Current Day Trading)
        # ============================================
        
        positions = await db["positions"].find({"userId": user_id}).to_list(None)
        
        if positions:
            positions_parts = []
            total_day_pnl = 0
            
            for p in positions:
                symbol = p.get("tradingsymbol", "UNKNOWN")
                bucket = p.get("bucket", "net")
                qty = p.get("qty", 0)
                pnl = p.get("pnl", 0)
                
                if bucket == "day":
                    total_day_pnl += pnl
                    positions_parts.append(
                        f"{symbol}: {qty} qty, Day P&L: ₹{pnl:,.2f}"
                    )
            
            if positions_parts:
                positions_breakdown = "\n".join(positions_parts)

                positions_summary = (
                    "CURRENT DAY POSITIONS:\n"
                    f"Total Day P&L: ₹{total_day_pnl:,.2f}\n"
                    "\n"
                    "Active Positions:\n"
                    f"{positions_breakdown}\n"
                    "\n"
                    f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
                )

                
                positions_vector = embed_text(positions_summary)

                await upsert_embedding(
                    db,
                    user_id=user_id,
                    kind="positions_summary",
                    doc_id=f"positions-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                    vector=positions_vector,
                    chunk=positions_summary,
                    metadata={
                        "dayPnL": total_day_pnl,
                        "positionsCount": len(positions_parts),
                        "generatedAt": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                embeddings_created += 1
                print(f"✅ Created positions summary embedding")
        
    except Exception as e:
        error_detail = traceback.format_exc()
        errors.append(str(e))
        print(f"[Embeddings] FULL ERROR for user {user_id}:")
        print(error_detail)
    
    return {
        "userId": str(user_id),
        "embeddingsCreated": embeddings_created,
        "errors": errors,
        "status": "success" if embeddings_created > 0 else "no_data",
        "breakdown": {
            "portfolio": 1 if holdings else 0,
            "tradeHistory": len(trades_by_month) if 'trades_by_month' in locals() else 0,
            "funds": 1 if funds else 0,
            "positions": 1 if positions and positions_parts else 0
        }
    }


async def list_active_users(db: AsyncIOMotorDatabase) -> List[ObjectId]:
    """
    Get list of all users with active Zerodha connections.
    Used by scheduler to sync all users periodically.
    """
    cursor = db["connections"].find(
        {"provider": "zerodha", "enabled": True},
        {"userId": 1}
    )
    
    user_ids = []
    async for doc in cursor:
        user_ids.append(doc["userId"])
    
    return user_ids