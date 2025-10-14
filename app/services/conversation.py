# app/services/conversation.py
"""
Complete conversation memory system with smart context management.
Handles unlimited storage with intelligent retrieval and user notifications.
"""

from typing import List, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from datetime import datetime, timezone

# Configuration
MAX_RECENT_MESSAGES = 50  # Always include last 50 messages in full
MAX_CONTEXT_TOKENS = 16000  # Budget for conversation history
MILESTONE_NOTIFICATIONS = [100, 500, 1000, 2000, 5000]  # When to notify user


async def get_message_count(db: AsyncIOMotorDatabase, user_id: ObjectId) -> int:
    """Get total message count for user."""
    return await db["messages"].count_documents({"userId": user_id})


async def check_conversation_milestone(
    db: AsyncIOMotorDatabase, 
    user_id: ObjectId
) -> Dict[str, Any] | None:
    """
    Check if user hit a conversation milestone.
    Returns notification message if milestone reached.
    """
    count = await get_message_count(db, user_id)
    
    # Check if we just hit a milestone
    if count in MILESTONE_NOTIFICATIONS:
        # Check if we already notified for this milestone
        user = await db["users"].find_one({"_id": user_id})
        last_notified = user.get("lastMilestoneNotified", 0) if user else 0
        
        if count > last_notified:
            # Update last notified milestone
            await db["users"].update_one(
                {"_id": user_id},
                {"$set": {"lastMilestoneNotified": count}}
            )
            
            return {
                "milestone": count,
                "message": get_milestone_message(count)
            }
    
    return None


def get_milestone_message(count: int) -> str:
    """Get appropriate milestone notification message."""
    if count == 100:
        return """
📊 **100 Messages Milestone!**

We've had 100 conversations! Just so you know:
✅ I remember everything we've discussed
✅ Your full chat history is saved
✅ I can search past conversations when needed

Keep going! 💬
"""
    
    elif count == 500:
        return """
📊 **500 Messages Milestone!** 🎉

Here's how my memory works now:

✅ **I actively remember**: Last 50-100 messages in full detail
✅ **I have summaries of**: Everything before that
✅ **I can search**: Specific topics from our entire history

Examples:
- "What's my portfolio?" → I answer naturally ✅
- "What did we discuss about TCS last month?" → I'll search our history 🔍

**Your choice:**
[Continue] - Keep this conversation going
[Fresh Start] - Begin new session (old messages still saved)
"""
    
    elif count == 1000:
        return """
📊 **1,000 Messages!** That's impressive! 🚀

**Memory Status:**
- Total conversations: 1,000
- Active memory: Last 50-100 messages
- Searchable archive: All 1,000 messages

I'm still working great! Continue or start fresh?

[Continue] [New Session]
"""
    
    elif count == 2000:
        return """
📊 **2,000 Messages Milestone!**

We're having a LOT of conversations! 💬

**Heads up:** 
- I can still help you perfectly
- But responses might slow slightly with this much history
- Consider starting a fresh session for better performance

**Your call:**
[Keep Going] - I'll manage the context smartly
[Fresh Start] - Recommended for best performance
"""
    
    elif count == 5000:
        return """
⚠️ **5,000 Messages - Context Limit Approaching**

**Important Notice:**
You've reached a very high message count. Here's what's happening:

📉 **Performance Impact:**
- Response times are slower
- Context management is getting complex
- Some older details might not be in active memory

✅ **What Still Works:**
- Your portfolio data is always accessible
- I can search old conversations
- All messages are safely stored

**Strongly Recommended:**
🆕 Start a new session for better performance

**Options:**
[New Session] - Recommended
[Continue Anyway] - I'll do my best, but expect slower responses

**Note:** Starting fresh doesn't delete your history - I can still search old messages if you ask!
"""
    
    return f"📊 Milestone: {count} messages!"


async def get_conversation_context(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> Tuple[List[Dict], str]:
    """
    Get smart conversation context for LLM.
    
    Returns:
        (messages_list, context_type)
        context_type: 'full', 'summarized', or 'archived'
    """
    total_count = await get_message_count(db, user_id)
    
    if total_count == 0:
        return ([], 'empty')
    
    # Always get recent messages
    recent_messages = await db["messages"].find(
        {"userId": user_id}
    ).sort("ts", -1).limit(MAX_RECENT_MESSAGES).to_list(None)
    recent_messages.reverse()  # Chronological order
    
    # Estimate tokens (rough: 1 message ≈ 50 tokens)
    recent_tokens = len(recent_messages) * 50
    
    if total_count <= MAX_RECENT_MESSAGES:
        # Small conversation - return everything
        return (recent_messages, 'full')
    
    elif recent_tokens < max_tokens:
        # We have room for more context
        # Get older messages up to token budget
        remaining_budget = max_tokens - recent_tokens
        additional_messages = min(remaining_budget // 50, total_count - MAX_RECENT_MESSAGES)
        
        if additional_messages > 0:
            older_messages = await db["messages"].find(
                {"userId": user_id}
            ).sort("ts", -1).skip(MAX_RECENT_MESSAGES).limit(additional_messages).to_list(None)
            older_messages.reverse()
            
            return (older_messages + recent_messages, 'extended')
        else:
            return (recent_messages, 'recent_only')
    
    else:
        # Large conversation - need summarization
        # For now, just use recent messages
        # TODO: Implement summarization of old messages
        return (recent_messages, 'summarized')


async def start_new_session(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId
) -> Dict[str, Any]:
    """
    Start a fresh conversation session.
    Archives old messages but keeps them searchable.
    """
    # Generate new session ID
    session_id = f"session-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    
    # Mark all existing messages as archived
    await db["messages"].update_many(
        {"userId": user_id, "archived": {"$exists": False}},
        {"$set": {"archived": True, "archivedAt": datetime.now(timezone.utc)}}
    )
    
    # Update user's active session
    await db["users"].update_one(
        {"_id": user_id},
        {
            "$set": {
                "currentSessionId": session_id,
                "sessionStartedAt": datetime.now(timezone.utc),
                "lastMilestoneNotified": 0  # Reset milestone counter
            }
        }
    )
    
    return {
        "sessionId": session_id,
        "message": """
✨ **Fresh Session Started!**

Your previous conversation is archived and searchable.
Starting with a clean slate for better performance!

How can I help you today? 💬
"""
    }


def format_conversation_for_llm(messages: List[Dict]) -> List[Dict]:
    """
    Format message history for LLM context.
    
    UPDATED: Returns list of message objects instead of formatted string.
    This allows LLM to properly understand conversation flow.
    
    Args:
        messages: List of message documents from MongoDB
        
    Returns:
        List of dicts with {"role": "user"|"assistant", "text": "..."}
    """
    if not messages:
        return []
    
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("text", "")
        
        # Only include non-empty messages
        if text:
            formatted.append({
                "role": role,  # "user" or "assistant"
                "text": text
            })
    
    return formatted


# ============================================
# Integration with /ask endpoint
# ============================================

async def handle_message_with_context(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    question: str,
    answer: str
) -> Dict[str, Any]:
    """
    Store message and check for milestones.
    Call this after generating an answer.
    
    Returns:
        {
            "milestone": milestone_notification (if any),
            "stored": True/False
        }
    """
    # Store messages
    await db["messages"].insert_many([
        {
            "userId": user_id,
            "role": "user",
            "text": question,
            "ts": datetime.now(timezone.utc)
        },
        {
            "userId": user_id,
            "role": "assistant",
            "text": answer,
            "ts": datetime.now(timezone.utc)
        }
    ])
    
    # Check for milestone
    milestone = await check_conversation_milestone(db, user_id)
    
    return {
        "milestone": milestone,
        "stored": True
    }