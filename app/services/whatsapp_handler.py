# app/services/whatsapp_handler.py
"""
WhatsApp handler with TradeChat-style login flow.
Matches the old working project exactly.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from typing import Dict, Any
import re

from app.services.user_management import get_or_create_user, update_user_preference
from app.services.retrieval import retrieve_context
from app.services.answer import answer_with_context
from app.services.conversation import (
    get_conversation_context,
    handle_message_with_context,
    format_conversation_for_llm,
    start_new_session
)
from app.services.websearch import web_search
from app.settings import settings


async def handle_whatsapp_message(
    db: AsyncIOMotorDatabase,
    phone: str,
    message: str,
    profile_name: str = "User"
) -> str:
    """
    Main handler for incoming WhatsApp messages.
    Replicates TradeChat flow exactly.
    
    Args:
        db: MongoDB database
        phone: User's phone number (format: 919876543210)
        message: Message text
        profile_name: User's WhatsApp profile name
    
    Returns:
        Response message to send back
    """
    # Get or create user (phone is the key)
    user = await get_or_create_user(db, phone)
    user_id = user["_id"]
    
    # Normalize message
    msg_lower = message.lower().strip()
    
    # ============================================
    # AUTHENTICATION FLOW (TradeChat Style)
    # ============================================
    
    # LOGIN COMMAND - Send Zerodha URL
    if msg_lower in ["login", "connect", "link"]:
        return await handle_login_command(phone)
    
    # DONE COMMAND - Complete authentication
    if msg_lower == "done":
        return await handle_done_command(db, phone, user_id)
    
    # ============================================
    # QUICK COMMANDS
    # ============================================
    
    if msg_lower in ["help", "/help", "?", "commands"]:
        return get_help_message(user)
    
    if msg_lower in ["portfolio", "p", "holdings"]:
        return await get_quick_portfolio(db, user_id, phone)
    
    if msg_lower in ["pnl", "profit", "loss"]:
        return await get_quick_pnl(db, user_id, phone)
    
    if msg_lower in ["status", "connected", "account"]:
        return await check_connection_status(db, user_id)
    
    # MODE SWITCHING
    if msg_lower.startswith("/mode ") or msg_lower in ["savage", "friendly", "professional", "funny"]:
        return await handle_mode_switch(db, phone, msg_lower)
    
    # NEW SESSION
    if msg_lower in ["new session", "/new", "fresh start", "reset"]:
        result = await start_new_session(db, user_id)
        return result["message"]
    
    # ============================================
    # PORTFOLIO QUESTIONS (RAG)
    # ============================================
    
    return await handle_question(db, user_id, phone, message)


# ============================================
# AUTHENTICATION HANDLERS (TradeChat Flow)
# ============================================

async def handle_login_command(phone: str) -> str:
    """
    Generate Zerodha login URL and send to user.
    Uses simple phone number as state (like old project).
    """
    # Generate Zerodha login URL with phone as state
    zerodha_url = (
        f"https://kite.zerodha.com/connect/login?"
        f"v=3&"
        f"api_key={settings.kite_api_key}&"
        f"state={phone}"  # Simple phone number, not encoded
    )
    
    return f"""🔐 **Connect Your Zerodha Account**

Click here to login: {zerodha_url}

After login:
1. Complete 2FA on Zerodha
2. You'll see a success page
3. Come back here and type *"done"*

🔒 Your credentials are secure and never stored."""


async def handle_done_command(
    db: AsyncIOMotorDatabase,
    phone: str,
    user_id: ObjectId
) -> str:
    """
    Complete authentication after user logs in.
    Claims any recent unused token (since Zerodha doesn't pass phone reliably).
    """
    from app.services.kite_client import exchange_request_token_for_access_token
    import datetime as dt
    from datetime import timezone
    
    # Get global temp tokens storage
    import app.main as main_app
    temp_tokens = getattr(main_app.app.state, 'temp_tokens', {})
    
    print(f"🔍 Looking for tokens. Total: {len(temp_tokens)}")
    
    # Find the most recent unused token (within 5 minutes)
    # Don't check phone since callback stores it as None
    request_token = None
    newest_timestamp = 0
    
    for token, data in temp_tokens.items():
        token_time = data.get('timestamp', 0)
        age = dt.datetime.now(timezone.utc).timestamp() - token_time
        is_used = data.get('used', False)
        
        print(f"  Token: {token[:20]}... | Used: {is_used} | Age: {age:.0f}s")
        
        if (not is_used and 
            age < 300 and  # Less than 5 minutes old
            token_time > newest_timestamp):
            # Found a newer unused token
            request_token = token
            newest_timestamp = token_time
    
    if not request_token:
        print(f"❌ No valid token found")
        return """❌ No recent login found.

Please type *"login"* and try again.

Make sure you:
1. Click the login link
2. Complete Zerodha login
3. Come back within 5 minutes
4. Type "done" """
    
    # Mark token as used and associate with this phone
    temp_tokens[request_token]['used'] = True
    temp_tokens[request_token]['phone'] = phone
    
    print(f"✅ Claiming token: {request_token[:20]}... for phone: {phone}")
    
    try:
        # Exchange request token for access token
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
        
        print(f"✅ Connected Zerodha for user {user_id}")
        
        # ============================================
        # TRIGGER IMMEDIATE SYNC AFTER LOGIN
        # ============================================
        
        try:
            from app.services.sync import run_incremental_sync, build_embeddings_for_user
            
            print(f"🔄 Starting portfolio sync...")
            
            # Sync portfolio data
            sync_result = await run_incremental_sync(
                db, 
                user_id,
                force_instruments=False,
                skip_embeddings=True
            )
            
            print(f"✅ Sync result: {sync_result}")
            
            # Generate embeddings
            print(f"🧠 Generating embeddings...")
            embeddings_result = await build_embeddings_for_user(db, user_id)
            
            print(f"✅ Embeddings: {embeddings_result}")
            
            # Success message with count
            holdings_count = await db["holdings"].count_documents({"userId": user_id})
            
            return f"""✅ **Successfully Connected!**

🎉 Your Zerodha account is linked!

📊 **Synced:**
- {holdings_count} holdings
- Portfolio data ready
- Embeddings generated

**Try:**
- "portfolio"
- "pnl"
- "How is my portfolio doing?"

What would you like to know?"""
        
        except Exception as sync_error:
            print(f"⚠️ Sync error: {sync_error}")
            import traceback
            traceback.print_exc()
            
            return """✅ **Successfully Connected!**

⚠️ Portfolio sync in progress...

Wait 30 seconds, then try:
- "portfolio"
- "pnl"

Or ask any question!"""
        
    except Exception as e:
        print(f"❌ Auth error: {e}")
        import traceback
        traceback.print_exc()
        return f"""❌ Authentication failed: {str(e)}

Please type *"login"* to try again."""

# ============================================
# PORTFOLIO QUERY HANDLERS
# ============================================

async def handle_question(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str,
    question: str
) -> str:
    """
    Handle a regular question using RAG.
    Checks authentication first.
    """
    # Get user data for preferences
    user = await db["users"].find_one({"_id": user_id})
    
    # Check if user has Zerodha connection
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        # Check if question is portfolio-related
        portfolio_keywords = [
            'portfolio', 'holdings', 'stocks', 'pnl', 'profit', 'loss',
            'performance', 'investment', 'value', 'cash', 'margin', 'trades'
        ]
        
        if any(keyword in question.lower() for keyword in portfolio_keywords):
            return """🔐 Please connect your Zerodha account first.

Type *"login"* to get started."""
    
    # Get conversation context
    conversation_messages, context_type = await get_conversation_context(
        db, user_id, max_tokens=16000
    )
    conversation_history = format_conversation_for_llm(conversation_messages)
    
    # Detect if question is about trade history
    q_low = question.lower()
    trade_keywords = [
        "buy", "bought", "sell", "sold", "trade", "purchase", "purchased",
        "september", "october", "november", "december", "january", "last month", 
        "last week", "recent", "history", "past", "did i", "when did",
        "sold any", "bought any"
    ]
    is_trade_query = any(kw in q_low for kw in trade_keywords)
    
    # Get portfolio context via vector search
    chunks = await retrieve_context(
        db,
        user_id=user_id,
        question=question,
        k=5,
        kind="portfolio_summary"
    )
    
    # ALSO search trade history if relevant
    if is_trade_query:
        print(f"🔍 Searching trade history for: {question}")
        trade_chunks = await retrieve_context(
            db,
            user_id=user_id,
            question=question,
            k=3,
            kind="trade_history"  # Search trade history embeddings!
        )
        print(f"📊 Found {len(trade_chunks)} trade history chunks")
        chunks = (chunks or []) + trade_chunks
    
    # Check if question needs web search
    if any(term in q_low for term in ["news", "market", "today", "latest", "current"]):
        web_query = question[7:].strip() if q_low.startswith("search:") else question
        web_chunks = web_search(web_query, k=3)
        for wc in web_chunks:
            wc["docId"] = f"{wc['docId']} (web)"
        chunks = (chunks or []) + web_chunks
    
    # Get user's personality mode (with fallback if user not found)
    persona = user.get("preferences", {}).get("personalityMode", "friendly") if user else "friendly"
    
    # Generate answer
    answer = answer_with_context(
        question,
        chunks,
        persona=persona,
        response_style="whatsapp",
        conversation_history=conversation_history
    )
    
    # Store conversation
    result = await handle_message_with_context(db, user_id, question, answer)
    
    # Add milestone notification if triggered
    if result.get("milestone"):
        milestone_msg = result["milestone"]["message"]
        answer = f"{answer}\n\n---\n{milestone_msg}"
    
    return answer


async def get_quick_portfolio(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str
) -> str:
    """Quick portfolio summary."""
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return """🔐 Please login first.

Type *"login"* to connect your Zerodha account."""
    
    # Get holdings from MongoDB
    holdings = await db["holdings"].find({"userId": user_id}).to_list(None)
    
    if not holdings:
        return """📊 No holdings found.

Your portfolio might be empty, or data hasn't synced yet.

Type *"refresh"* to sync your data."""
    
    # Calculate totals
    total_value = 0
    total_investment = 0
    
    for h in holdings:
        qty = h.get("qty", 0)
        avg_price = h.get("avgPrice", 0)
        last_price = h.get("lastPrice", 0)
        
        total_investment += qty * avg_price
        total_value += qty * last_price
    
    total_pnl = total_value - total_investment
    pnl_pct = (total_pnl / total_investment * 100) if total_investment > 0 else 0
    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    
    response = f"""📊 **Your Portfolio**

💼 Holdings: {len(holdings)} stocks
💰 Total Value: ₹{total_value:,.2f}
{pnl_emoji} P&L: ₹{total_pnl:,.2f} ({pnl_pct:+.2f}%)

💡 Ask me about any specific stock!"""
    
    return response


async def get_quick_pnl(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str
) -> str:
    """Quick P&L summary."""
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return """🔐 Please login first.

Type *"login"* to connect your Zerodha account."""
    
    # Get holdings
    holdings = await db["holdings"].find({"userId": user_id}).to_list(None)
    
    if not holdings:
        return "📊 No holdings found. Type *refresh* to sync."
    
    # Calculate P&L
    stocks_pnl = []
    total_pnl = 0
    
    for h in holdings:
        symbol = h.get("tradingsymbol", "UNKNOWN")
        qty = h.get("qty", 0)
        avg_price = h.get("avgPrice", 0)
        last_price = h.get("lastPrice", 0)
        
        investment = qty * avg_price
        current_value = qty * last_price
        pnl = current_value - investment
        pnl_pct = (pnl / investment * 100) if investment > 0 else 0
        
        total_pnl += pnl
        stocks_pnl.append({
            "symbol": symbol,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
    
    # Sort by P&L
    stocks_pnl.sort(key=lambda x: x["pnl"], reverse=True)
    
    top_gainers = [s for s in stocks_pnl if s["pnl"] > 0][:3]
    top_losers = [s for s in stocks_pnl if s["pnl"] < 0][-3:]
    
    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    
    response = f"""{pnl_emoji} **P&L Summary**

Total P&L: ₹{total_pnl:,.2f}

"""
    
    if top_gainers:
        response += "🚀 **Top Gainers:**\n"
        for s in top_gainers:
            response += f"• {s['symbol']}: +₹{s['pnl']:,.2f} ({s['pnl_pct']:+.2f}%)\n"
    
    if top_losers:
        response += "\n📉 **Underperformers:**\n"
        for s in top_losers:
            response += f"• {s['symbol']}: ₹{s['pnl']:,.2f} ({s['pnl_pct']:+.2f}%)\n"
    
    return response


async def check_connection_status(db: AsyncIOMotorDatabase, user_id: ObjectId) -> str:
    """Check Zerodha connection status."""
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return """❌ **Not Connected**

You haven't linked your Zerodha account yet.

Type *"login"* to connect now!"""
    
    return """✅ **Connected to Zerodha**

Your account is linked and active!

Available commands:
• *portfolio* - View your holdings
• *pnl* - Check profit/loss

Or just ask me anything!"""


# ============================================
# MODE SWITCHING
# ============================================

async def handle_mode_switch(
    db: AsyncIOMotorDatabase,
    phone: str,
    message: str
) -> str:
    """Handle personality mode switching."""
    # Extract mode
    mode = message.replace("/mode ", "").strip()
    
    valid_modes = {
        "savage": "💀 Savage mode activated. Prepare for brutal honesty.",
        "friendly": "😊 Friendly mode activated! Let's chat casually.",
        "professional": "📊 Professional mode activated. Formal analysis engaged.",
        "funny": "🤣 Funny mode activated! Time for memes! 🚀"
    }
    
    if mode not in valid_modes:
        return f"""Choose a mode:
• *savage* - Brutally honest 💀
• *friendly* - Casual & warm 😊  
• *professional* - Formal analysis 📊
• *funny* - Memes & humor 🤣

Example: Type "savage" or "/mode savage" """
    
    # Update user preference
    success = await update_user_preference(db, phone, "personalityMode", mode)
    
    if success:
        return valid_modes[mode]
    else:
        return "❌ Failed to switch mode. Try again."


# ============================================
# HELP MESSAGE
# ============================================

def get_help_message(user: Dict) -> str:
    """Return help message with all commands."""
    has_connection = user.get("zerodhaAuth", {}).get("isAuthenticated", False)
    
    if has_connection:
        return """🤖 **TradeBuddy Commands**

**Portfolio:**
• `portfolio` - View holdings
• `pnl` - Profit/loss summary
• `status` - Connection status

**Modes:**
• `savage` - Brutally honest 💀
• `friendly` - Casual 😊
• `professional` - Formal 📊
• `funny` - Memes 🤣

**Natural Questions:**
• "How's my portfolio?"
• "Should I buy TCS?"
• "Top gainers today?"

Type any question naturally!"""
    
    else:
        return """🤖 **TradeBuddy Help**

🔐 **First time?**
Type *"login"* to connect your Zerodha account

**After connecting:**
• View portfolio & P&L
• Ask questions naturally
• Get personalized advice

Type *"login"* to get started!"""