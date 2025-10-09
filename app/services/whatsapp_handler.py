# app/services/whatsapp_handler.py
"""
Main logic for handling WhatsApp messages and commands.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from typing import Dict, Any

from app.services.user_management import get_or_create_user, update_user_preference
from app.services.retrieval import retrieve_context
from app.services.answer import answer_with_context
from app.services.conversation import (
    get_conversation_context,
    handle_message_with_context,
    format_conversation_for_llm
)


async def handle_whatsapp_message(
    db: AsyncIOMotorDatabase,
    phone: str,
    message: str,
    profile_name: str = "User"
) -> str:
    """
    Main handler for incoming WhatsApp messages.
    Routes to appropriate command or query handler.
    
    Args:
        db: MongoDB database
        phone: User's phone number
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
    # COMMANDS
    # ============================================
    
    # Help command
    if msg_lower in ["help", "/help", "?", "commands"]:
        return get_help_message()
    
    # Mode switching
    if msg_lower.startswith("/mode "):
        mode = msg_lower.replace("/mode ", "").strip()
        return await handle_mode_switch(db, phone, mode)
    
    # Refresh portfolio
    if msg_lower in ["refresh", "/refresh", "sync", "/sync"]:
        return await handle_refresh_command(db, user_id)
    
    # New session
    if msg_lower in ["new session", "/new", "fresh start", "reset"]:
        from app.services.conversation import start_new_session
        result = await start_new_session(db, user_id)
        return result["message"]
    
    # ============================================
    # NORMAL CONVERSATION (RAG)
    # ============================================
    
    return await handle_question(db, user_id, phone, message)


async def handle_question(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    phone: str,
    question: str
) -> str:
    """
    Handle a regular question using RAG.
    """
    # Get conversation context
    conversation_messages, context_type = await get_conversation_context(
        db, user_id, max_tokens=16000
    )
    conversation_history = format_conversation_for_llm(conversation_messages)
    
    # Get portfolio context
    chunks = await retrieve_context(
        db,
        user_id=user_id,
        question=question,
        k=5,
        kind="portfolio_summary"
    )
    
    # Get user's personality mode (default: savage)
    user = await db["users"].find_one({"_id": user_id})
    persona = user.get("preferences", {}).get("personalityMode", "savage")
    
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


async def handle_mode_switch(
    db: AsyncIOMotorDatabase,
    phone: str,
    mode: str
) -> str:
    """Handle personality mode switching."""
    valid_modes = ["professional", "friendly", "funny", "savage"]
    
    if mode not in valid_modes:
        return f"❌ Invalid mode. Choose: {', '.join(valid_modes)}"
    
    success = await update_user_preference(db, phone, "personalityMode", mode)
    
    if success:
        responses = {
            "professional": "📊 Professional mode activated. Formal analysis mode engaged.",
            "friendly": "😊 Friendly mode activated! Back to chill conversations.",
            "funny": "🤣 FUNNY MODE ACTIVATED! Let's make trading entertaining! 🚀",
            "savage": "💀 SAVAGE MODE UNLOCKED. Prepare for brutal honesty. No feelings spared."
        }
        return responses[mode]
    else:
        return "❌ Failed to switch mode. Try again."


async def handle_refresh_command(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId
) -> str:
    """Handle portfolio refresh command."""
    # Check if user has Zerodha connection
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return """
❌ No Zerodha connection found!

Connect your account first:
1. Go to: http://your-app-url/zerodha/connect
2. Enter your phone number
3. Login to Zerodha
4. Come back here and type 'refresh'

Or use the quick connect command:
/connect
"""
    
    # Trigger sync (this would call your scheduler)
    # For now, return a message
    return """
🔄 Refreshing your portfolio data...

This will take 10-15 seconds. Fetching:
✅ Holdings
✅ Trades
✅ Positions
✅ Funds
✅ Orders

Type 'done' when ready or ask any question!
"""


def get_help_message() -> str:
    """Return help message with all commands."""
    return """
🤖 **TradeBuddy Commands**

**Portfolio:**
- `refresh` - Sync latest data from Zerodha
- `portfolio` - Show your portfolio summary
- `holdings` - List all your stocks

**Modes:**
- `/mode savage` - Brutally honest (default) 💀
- `/mode funny` - Entertaining Grok-style 🤣
- `/mode friendly` - Casual & supportive 😊
- `/mode professional` - Formal analysis 📊

**Conversation:**
- `new session` - Start fresh conversation
- `help` - Show this message

**Just Ask:**
Ask any financial question naturally!
Examples:
- "How's my portfolio?"
- "Should I buy gold?"
- "What are tech stock trends?"
- "How much cash do I have?"

💀 **Default Mode: SAVAGE**
I'll roast your portfolio but give real advice!
"""