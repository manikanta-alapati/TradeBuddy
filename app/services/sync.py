# app/services/sync.py
from __future__ import annotations
import datetime as dt
from datetime import timezone
import traceback

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
    Complete embeddings generation pipeline:
    1. Generate portfolio summary
    2. Generate symbol-month summaries
    3. Convert to embeddings
    4. Store in MongoDB
    
    This makes your portfolio data searchable with semantic queries.
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
                
                total_investment += investment
                total_value += current_value
                
                # Create human-readable text for this holding
                holdings_text_parts.append(
                    f"{symbol}: {qty} shares at avg Rs{avg_price:.2f}, "
                    f"current Rs{last_price:.2f}, "
                    f"P&L: Rs{pnl:,.2f} ({pnl_pct:+.2f}%)"
                )
            
            # Overall portfolio summary
            total_pnl = total_value - total_investment
            total_pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
            holdings_breakdown = "\n".join(holdings_text_parts)

            portfolio_summary = (
                "Portfolio Overview:\n"
                f"Total Holdings: {len(holdings)} stocks\n"
                f"Total Investment: Rs.{total_investment:,.2f}\n"
                f"Current Value: Rs.{total_value:,.2f}\n"
                f"Total P&L: Rs.{total_pnl:,.2f} ({total_pnl_pct:+.2f}%)\n"
                "\n"
                "Holdings Breakdown:\n"
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
        
        # ============================================
        # 2. FUNDS SUMMARY EMBEDDING
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
                    f"{segment}: Available Rs{available:,.2f}, Net Rs{net:,.2f}"
                )
            
            funds_breakdown = "\n".join(funds_parts)

            funds_summary = (
                "Account Funds:\n"
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
        
        # ============================================
        # 3. SYMBOL-MONTH SUMMARIES (Per Stock Performance)
        # ============================================
        
        # Get last 3 months of trade data
        from datetime import timedelta
        three_months_ago = datetime.now(timezone.utc) - timedelta(days=90)
        
        # Get unique symbols that were traded
        trades = await db["trades"].find({
            "userId": user_id,
            "ts": {"$gte": three_months_ago}
        }).to_list(None)
        
        if trades:
            # Group trades by symbol
            symbol_trades = {}
            for trade in trades:
                symbol = trade.get("symbol", "UNKNOWN")
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)
            
            # Create summary for each symbol
            for symbol, trades_list in symbol_trades.items():
                # Calculate metrics
                buy_amount = 0
                sell_amount = 0
                total_qty = 0
                
                for t in trades_list:
                    qty = t.get("qty", 0)
                    price = t.get("price", 0)
                    amount = qty * price
                    
                    # Determine if buy or sell (you may need to add 'side' field to trades)
                    # For now, we'll just aggregate all trades
                    total_qty += qty
                    buy_amount += amount
                
                # Get current holding for this symbol
                holding = await db["holdings"].find_one({
                    "userId": user_id,
                    "tradingsymbol": symbol
                })
                
                current_price = holding.get("lastPrice", 0) if holding else 0
                current_qty = holding.get("qty", 0) if holding else 0
                
                symbol_summary = (
                    f"{symbol} Trading Activity (Last 3 Months):\n"
                    f"Total Trades: {len(trades_list)}\n"
                    f"Total Quantity Traded: {total_qty}\n"
                    f"Current Holding: {current_qty} shares at Rs.{current_price:.2f}\n"
                    f"Trade Volume: Rs.{buy_amount:,.2f}\n"
                    "\n"
                    f"Recent Activity: {len(trades_list)} trades executed in the last 90 days.\n"
                    "\n"
                    f"Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)
                
                symbol_vector = embed_text(symbol_summary)

                await upsert_embedding(
                    db,
                    user_id=user_id,
                    kind="symbol_summary",
                    doc_id=f"{symbol}-recent",
                    vector=symbol_vector,
                    chunk=symbol_summary,
                    metadata={
                        "symbol": symbol,
                        "tradesCount": len(trades_list),
                        "generatedAt": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                embeddings_created += 1
        
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
                        f"{symbol}: {qty} qty, Day P&L: Rs{pnl:,.2f}"
                    )
            
            if positions_parts:
                positions_breakdown = "\n".join(positions_parts)

                positions_summary = (
                    "Current Day Positions:\n"
                    f"Total Day P&L: Rs.{total_day_pnl:,.2f}\n"
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
        
    except Exception as e:
        error_detail = traceback.format_exc()
        errors.append(str(e))
        print(f"[Embeddings] FULL ERROR for user {user_id}:")
        print(error_detail)
    
    return {
        "userId": str(user_id),
        "embeddingsCreated": embeddings_created,
        "errors": errors,
        "status": "success" if embeddings_created > 0 else "no_data"
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