# app/services/whatsapp_handler.py - COMPLETE FIX

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from typing import Dict, Any

from app.services.user_management import get_or_create_user, update_user_preference
from app.services.retrieval import retrieve_context
from app.services.answer import answer_with_context
from app.services.conversation import (
    get_conversation_context,
    handle_message_with_context,
    format_conversation_for_llm,
    start_new_session
)
from app.settings import settings


# app/services/whatsapp_handler.py - UPDATE

async def handle_whatsapp_message(
    db: AsyncIOMotorDatabase,
    phone: str,
    message: str,
    profile_name: str = "User"
) -> str:
    """Main handler - FIXED VERSION"""
    
    user = await get_or_create_user(db, phone)
    user_id = user["_id"]
    
    msg_lower = message.lower().strip()
    
    print(f"🔍 [HANDLER] Processing: '{message}' (normalized: '{msg_lower}')")
    
    # Get connection status
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    # ============================================
    # PRIORITY 1: ALL COMMANDS
    # ============================================
    
    # LOGIN
    if msg_lower in ["login", "connect", "link"]:
        print(f"✅ [HANDLER] Matched LOGIN command")
        return await handle_login_command(phone)
    
    # DONE
    if msg_lower == "done":
        print(f"✅ [HANDLER] Matched DONE command")
        return await handle_done_command(db, phone, user_id)
    
    # PORTFOLIO
    if msg_lower in ["portfolio", "p", "holdings"]:
        print(f"📈 [HANDLER] Matched PORTFOLIO command")
        if not conn:
            return "You haven't connected your Zerodha account yet!\n\nType \"login\" to get started."
        return await get_quick_portfolio(db, user_id, phone)
    
    # PNL
    if msg_lower in ["pnl", "profit", "loss"]:
        print(f"💰 [HANDLER] Matched PNL command")
        if not conn:
            return "You haven't connected your Zerodha account yet!\n\nType \"login\" to get started."
        return await get_quick_pnl(db, user_id, phone)
    
    # HELP
    if msg_lower in ["help", "/help", "?", "commands", "start"]:
        print(f"✅ [HANDLER] Matched HELP command")
        return get_help_message(user, conn is not None)
    
    # MODES
    if msg_lower in ["modes", "personas", "personalities", "styles"]:
        print(f"✅ [HANDLER] Matched MODES command")
        return get_personas_message()
    
    # STATUS
    if msg_lower in ["status", "connected", "account"]:
        print(f"✅ [HANDLER] Matched STATUS command")
        return await check_connection_status(db, user_id)
    
    # SWITCH MODE
    if msg_lower in ["savage", "friendly", "professional", "funny"]:
        print(f"✅ [HANDLER] Matched MODE SWITCH command")
        return await handle_mode_switch(db, phone, msg_lower)
    
    # NEW SESSION
    if msg_lower in ["new session", "/new", "fresh start", "reset chat"]:
        print(f"✅ [HANDLER] Matched NEW SESSION command")
        result = await start_new_session(db, user_id)
        return result["message"]
    
    # REFRESH (manual sync
    if msg_lower in ["refresh", "sync", "update"]:
        print(f"🔄 [HANDLER] Matched REFRESH command")
        if not conn:
            return "You haven't connected your Zerodha account yet!\n\nType \"login\" to get started."
        return await handle_refresh_command(db, user_id)
    
          
    
    # ============================================
    # PRIORITY 2: REGULAR QUESTIONS
    # ============================================
    
    # Check if user needs to connect first
    if not conn:
        portfolio_keywords = ['portfolio', 'holdings', 'stocks', 'pnl', 'profit', 'loss', 'tatacap', 'tcs', 'reliance']
        if any(keyword in msg_lower for keyword in portfolio_keywords):
            # Check if first time user (no messages yet)
            message_count = await db["messages"].count_documents({"userId": user_id})
            if message_count == 0:
                print(f"👋 [HANDLER] First message from unconnected user - showing welcome")
                return get_welcome_message(profile_name)
            else:
                return "Please connect your Zerodha account first.\n\nType \"login\" to get started."
    
    # Process as regular question (this will save the message!)
    print(f"💬 [HANDLER] Processing as regular question")
    return await handle_question(db, user_id, phone, message)


# ============================================
# WELCOME & INFO MESSAGES
# ============================================

def get_welcome_message(profile_name: str) -> str:
    """Welcome message for first-time users."""
    return f"""Hey {profile_name}! Welcome to TradeBuddy!

I'm your AI-powered personal financial advisor. I can help you:

- Check your portfolio anytime
- Track P&L and performance
- Get stock recommendations
- Search latest market news
- Answer investment questions

HOW TO GET STARTED:

Step 1: Connect Your Zerodha Account
Type: login

I'll send you a secure link to connect. Your credentials are never stored!

Step 2: Choose Your Style
I have 4 personalities:
- Friendly (casual & warm)
- Professional (formal analysis)
- Savage (brutally honest)
- Funny (memes & humor)

Type "modes" to learn more!

Step 3: Start Chatting
Once connected, just ask me anything:
- "What's my portfolio?"
- "Should I buy TCS?"
- "Latest tech stock news?"

Ready to connect? Type: login"""


def get_personas_message() -> str:
    """Explain the 4 personality modes."""
    return """Choose Your TradeBuddy Personality:

1. FRIENDLY (Default)
Casual, warm, and conversational
Perfect for everyday questions

2. PROFESSIONAL
Formal, analytical, data-driven
Best for serious analysis

3. SAVAGE
Brutally honest, no sugarcoating
For traders who want the hard truth

4. FUNNY
Memes, jokes, casual vibes
Make investing fun

TO SWITCH MODES:
Type: friendly
Type: professional
Type: savage
Type: funny

Current questions? Just ask away!"""


def get_help_message(user: dict, has_connection: bool) -> str:
    """Help message - different based on connection status."""
    if has_connection:
        current_mode = user.get("preferences", {}).get("personalityMode", "friendly")
        
        return f"""TradeBuddy Help

QUICK COMMANDS:
- portfolio - View your holdings
- pnl - Check profit/loss
- status - Check connection
- modes - Change personality

PERSONALITY MODES:
Current: {current_mode.title()}
Switch: Type mode name

NATURAL QUESTIONS:
Just ask naturally!

NEW SESSION:
- Type: new session

Need help? Just ask!"""
    
    else:
        return """TradeBuddy Help

You haven't connected your Zerodha account yet!

TO GET STARTED:
1. Type: login
2. Connect Zerodha (secure)
3. Type: done
4. Start asking!

Ready? Type: login"""


# ============================================
# AUTHENTICATION HANDLERS
# ============================================

async def handle_login_command(phone: str) -> str:
    """Generate Zerodha login URL."""
    zerodha_url = (
        f"https://kite.zerodha.com/connect/login?"
        f"v=3&"
        f"api_key={settings.kite_api_key}&"
        f"state={phone}"
    )
    
    return f"""Connect Your Zerodha Account

Click here to login: {zerodha_url}

After login:
1. Complete 2FA on Zerodha
2. You'll see a success page
3. Come back here and type "done"

Your credentials are secure and never stored."""


async def handle_done_command(
    db: AsyncIOMotorDatabase,
    phone: str,
    user_id: ObjectId
) -> str:
    """Complete authentication after user logs in."""
    from app.services.kite_client import exchange_request_token_for_access_token
    import datetime as dt
    from datetime import timezone
    
    # Get global temp tokens storage
    import app.main as main_app
    temp_tokens = getattr(main_app.app.state, 'temp_tokens', {})
    
    print(f"🔍 Looking for tokens. Total: {len(temp_tokens)}")
    
    # Find the most recent unused token
    request_token = None
    newest_timestamp = 0
    
    for token, data in temp_tokens.items():
        token_time = data.get('timestamp', 0)
        age = dt.datetime.now(timezone.utc).timestamp() - token_time
        is_used = data.get('used', False)
        
        if (not is_used and age < 300 and token_time > newest_timestamp):
            request_token = token
            newest_timestamp = token_time
    
    if not request_token:
        return """No recent login found.

Please type "login" and try again."""
    
    # Mark token as used
    temp_tokens[request_token]['used'] = True
    temp_tokens[request_token]['phone'] = phone
    
    try:
        # Exchange token
        tokens = exchange_request_token_for_access_token(request_token)
        
        # Save to database
        await db["connections"].update_one(
            {"userId": user_id, "provider": "zerodha"},
            {"$set": {
                "userId": user_id,
                "provider": "zerodha",
                "apiKey": settings.kite_api_key,
                "apiSecret": settings.kite_api_secret,
                "accessToken": tokens["access_token"],
                "publicToken": tokens.get("public_token"),
                "scopes": ["read"],
                "enabled": True,
                "createdAt": dt.datetime.now(timezone.utc),
            }},
            upsert=True
        )
        
        # Trigger sync
        try:
            from app.services.sync import run_incremental_sync, build_embeddings_for_user
            
            await run_incremental_sync(db, user_id, force_instruments=False, skip_embeddings=True)
            await build_embeddings_for_user(db, user_id)
            
            holdings_count = await db["holdings"].count_documents({"userId": user_id})
            
            return f"""Successfully Connected!

Your Zerodha account is linked!

Synced:
- {holdings_count} holdings
- Portfolio data ready

Try:
- "portfolio"
- "pnl"
- "How is my portfolio doing?"

What would you like to know?"""
        
        except Exception as sync_error:
            return """Successfully Connected!

Portfolio sync in progress...

Wait 30 seconds, then try:
- "portfolio"
- "pnl" """
        
    except Exception as e:
        return f"""Authentication failed: {str(e)}

Please type "login" to try again."""


# ============================================
# PORTFOLIO COMMANDS
# ============================================

# app/services/whatsapp_handler.py - UPDATE get_quick_portfolio

async def get_quick_portfolio(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str
) -> str:
    """Quick portfolio summary - UPDATED with positions."""
    
    # Get holdings (long-term)
    holdings = await db["holdings"].find({"userId": user_id}).to_list(None)
    
    # Get positions (active trades)
    positions = await db["positions"].find({"userId": user_id}).to_list(None)
    
    if not holdings and not positions:
        return """No holdings or positions found.

Your portfolio might be empty.

Type "refresh" to sync latest data."""
    
    # Calculate holdings totals
    holdings_value = 0
    holdings_investment = 0
    
    for h in holdings:
        qty = h.get("qty", 0)
        avg_price = h.get("avgPrice", 0)
        last_price = h.get("lastPrice", 0)
        
        holdings_investment += qty * avg_price
        holdings_value += qty * last_price
    
    # Calculate positions totals
    positions_value = 0
    positions_investment = 0
    
    for p in positions:
        qty = p.get("quantity", 0)
        avg_price = p.get("average_price", 0)
        last_price = p.get("last_price", 0)
        
        positions_investment += qty * avg_price
        positions_value += qty * last_price
    
    # Total P&L
    total_investment = holdings_investment + positions_investment
    total_value = holdings_value + positions_value
    total_pnl = total_value - total_investment
    pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    
    response = f"""Your Portfolio

HOLDINGS: {len(holdings)} stocks
Value: Rs.{holdings_value:,.2f}

POSITIONS: {len(positions)} trades
Value: Rs.{positions_value:,.2f}

TOTAL VALUE: Rs.{total_value:,.2f}
P&L: Rs.{total_pnl:,.2f} ({pnl_pct:+.2f}%)

Ask me about any stock or position!"""
    
    return response


# app/services/whatsapp_handler.py - UPDATE get_quick_pnl

async def get_quick_pnl(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str
) -> str:
    """Quick P&L summary - UPDATED with positions."""
    
    # Get holdings
    holdings = await db["holdings"].find({"userId": user_id}).to_list(None)
    
    # Get positions
    positions = await db["positions"].find({"userId": user_id}).to_list(None)
    
    if not holdings and not positions:
        return "No holdings or positions found."
    
    stocks_pnl = []
    
    # Calculate P&L for holdings
    for h in holdings:
        symbol = h.get("tradingsymbol", "UNKNOWN")
        qty = h.get("qty", 0)
        avg_price = h.get("avgPrice", 0)
        last_price = h.get("lastPrice", 0)
        
        investment = qty * avg_price
        current_value = qty * last_price
        pnl = current_value - investment
        pnl_pct = (pnl / investment * 100) if investment > 0 else 0
        
        stocks_pnl.append({
            "symbol": symbol,
            "type": "HOLDING",
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
    
    # Calculate P&L for positions
    for p in positions:
        symbol = p.get("tradingsymbol", "UNKNOWN")
        qty = p.get("quantity", 0)
        avg_price = p.get("average_price", 0)
        last_price = p.get("last_price", 0)
        
        investment = qty * avg_price
        current_value = qty * last_price
        pnl = current_value - investment
        pnl_pct = (pnl / investment * 100) if investment > 0 else 0
        
        stocks_pnl.append({
            "symbol": symbol,
            "type": "POSITION",
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
    
    # Sort by P&L
    stocks_pnl.sort(key=lambda x: x["pnl"], reverse=True)
    
    # Calculate totals
    total_pnl = sum(s["pnl"] for s in stocks_pnl)
    
    top_gainers = [s for s in stocks_pnl if s["pnl"] > 0][:5]
    top_losers = [s for s in stocks_pnl if s["pnl"] < 0][-5:]
    
    response = f"""P&L Summary

Total P&L: Rs.{total_pnl:,.2f}

"""
    
    if top_gainers:
        response += "Top Gainers:\n"
        for s in top_gainers:
            response += f"- {s['symbol']} ({s['type']}): +Rs.{s['pnl']:,.2f} ({s['pnl_pct']:+.2f}%)\n"
    
    if top_losers:
        response += "\nUnderperformers:\n"
        for s in top_losers:
            response += f"- {s['symbol']} ({s['type']}): Rs.{s['pnl']:,.2f} ({s['pnl_pct']:+.2f}%)\n"
    
    response += f"\nHoldings: {len(holdings)} | Positions: {len(positions)}"
    
    return response


async def check_connection_status(db: AsyncIOMotorDatabase, user_id: ObjectId) -> str:
    """Check Zerodha connection status."""
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return """Not Connected

You haven't linked your Zerodha account yet.

Type "login" to connect now!"""
    
    return """Connected to Zerodha

Your account is linked and active!

Available commands:
- portfolio - View holdings
- pnl - Check profit/loss

Or just ask me anything!"""


async def handle_mode_switch(
    db: AsyncIOMotorDatabase,
    phone: str,
    message: str
) -> str:
    """Handle personality mode switching."""
    mode = message.strip()
    
    valid_modes = {
        "savage": "Savage mode activated. Prepare for brutal honesty.",
        "friendly": "Friendly mode activated! Let's chat casually.",
        "professional": "Professional mode activated. Formal analysis engaged.",
        "funny": "Funny mode activated! Time for memes!"
    }
    
    if mode not in valid_modes:
        return """Choose a mode:
- savage
- friendly
- professional
- funny

Type the mode name to switch."""
    
    success = await update_user_preference(db, phone, "personalityMode", mode)
    
    return valid_modes[mode] if success else "Failed to switch mode. Try again."


# ============================================
# REGULAR QUESTIONS (RAG + LLM)
# ============================================

async def handle_question(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str,
    question: str
) -> str:
    """Handle regular question using RAG."""
    user = await db["users"].find_one({"_id": user_id})
    
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        portfolio_keywords = ['portfolio', 'holdings', 'stocks', 'pnl', 'profit', 'loss']
        if any(keyword in question.lower() for keyword in portfolio_keywords):
            return """Please connect your Zerodha account first.

Type "login" to get started."""
    
    conversation_messages, _ = await get_conversation_context(db, user_id, max_tokens=16000)
    conversation_history = format_conversation_for_llm(conversation_messages)
    
    chunks = await retrieve_context(db, user_id=user_id, question=question, k=5, kind="portfolio_summary")
    
    persona = user.get("preferences", {}).get("personalityMode", "friendly") if user else "friendly"
    
    answer = answer_with_context(
        question,
        chunks,
        persona=persona,
        response_style="whatsapp",
        conversation_history=conversation_history
    )
    
    await handle_message_with_context(db, user_id, question, answer)
    
    return answer





# In app/services/whatsapp_handler.py - Replace handle_refresh_command

async def handle_refresh_command(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId
) -> str:
    """
    Manually trigger portfolio sync with token validation.
    FIXED: Properly checks token validity and pulls fresh data.
    """
    from app.services.kite_client import build_kite_client
    from app.services.mappers import map_holdings, map_positions, map_funds
    from app.services.sync import build_embeddings_for_user
    from datetime import datetime, timezone
    
    print(f"🔄 [REFRESH] Starting manual refresh for user {user_id}")
    
    # Get connection
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return """❌ Zerodha not connected.

Type "login" to connect your account."""
    
    # Build Kite client
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    
    try:
        # CRITICAL: Test token validity first
        print(f"🔐 [REFRESH] Testing token validity...")
        profile = kite.profile()  # This will fail if token expired
        print(f"✅ [REFRESH] Token valid for user: {profile.get('user_name')}")
        
    except Exception as e:
        print(f"❌ [REFRESH] Token expired or invalid: {e}")
        
        # Clear invalid connection
        await db["connections"].update_one(
            {"userId": user_id, "provider": "zerodha"},
            {"$set": {"enabled": False, "tokenExpiredAt": datetime.now(timezone.utc)}}
        )
        
        return """❌ Zerodha session expired!

Your login token has expired. Zerodha tokens are only valid for 1 day.

Type "login" to reconnect your account."""
    
    try:
        # Token is valid, proceed with refresh
        print(f"📊 [REFRESH] Fetching fresh data from Zerodha...")
        
        # 1. Get fresh holdings
        raw_holdings = kite.get_holdings()
        print(f"📈 [REFRESH] Got {len(raw_holdings)} holdings")
        
        # 2. Get fresh positions
        raw_positions = kite.get_positions()
        print(f"📊 [REFRESH] Got positions data")
        
        # 3. Get fresh funds
        raw_funds = kite.get_funds()
        print(f"💰 [REFRESH] Got funds data")
        
        # DELETE old data before inserting new
        print(f"🗑️ [REFRESH] Clearing old data...")
        await db["holdings"].delete_many({"userId": user_id})
        await db["positions"].delete_many({"userId": user_id})
        await db["funds"].delete_many({"userId": user_id})
        
        # Map and insert fresh data
        holdings = map_holdings(raw_holdings)
        for h in holdings:
            h["userId"] = user_id
            h["syncedAt"] = datetime.now(timezone.utc)
        
        if holdings:
            await db["holdings"].insert_many(holdings)
            print(f"✅ [REFRESH] Inserted {len(holdings)} holdings")
        
        # Insert positions
        positions = map_positions(raw_positions)
        for p in positions:
            p["userId"] = user_id
            p["syncedAt"] = datetime.now(timezone.utc)
        
        if positions:
            await db["positions"].insert_many(positions)
            print(f"✅ [REFRESH] Inserted {len(positions)} positions")
        
        # Insert funds
        funds = map_funds(raw_funds)
        for f in funds:
            f["userId"] = user_id
            f["syncedAt"] = datetime.now(timezone.utc)
        
        if funds:
            await db["funds"].insert_many(funds)
            print(f"✅ [REFRESH] Inserted {len(funds)} fund records")
        
        # Calculate summary
        total_value = sum(h.get("qty", 0) * h.get("lastPrice", 0) for h in holdings)
        total_investment = sum(h.get("qty", 0) * h.get("avgPrice", 0) for h in holdings)
        total_pnl = total_value - total_investment
        pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
        
        # Regenerate embeddings in background
        print(f"🤖 [REFRESH] Regenerating embeddings...")
        try:
            await build_embeddings_for_user(db, user_id)
            embeddings_status = "✅"
        except Exception as e:
            print(f"⚠️ [REFRESH] Embeddings generation failed: {e}")
            embeddings_status = "⚠️"
        
        print(f"✅ [REFRESH] Complete!")
        
        return f"""✅ Portfolio Refreshed!

📊 Synced: {len(holdings)} holdings
💰 Value: ₹{total_value:,.2f}
📈 P&L: ₹{total_pnl:,.2f} ({pnl_pct:+.2f}%)
🕐 Updated: Just now
{embeddings_status} AI Memory: Updated

Your data is now live from Zerodha!"""
        
    except Exception as e:
        print(f"❌ [REFRESH] Sync error: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if it's a specific Zerodha error
        error_str = str(e).lower()
        
        if "token" in error_str or "session" in error_str:
            return """❌ Session expired!

Type "login" to reconnect your Zerodha account."""
        
        return f"""❌ Refresh failed!

Error: {str(e)}

Try again or type "login" to reconnect."""