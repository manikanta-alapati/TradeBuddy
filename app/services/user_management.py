"""
Step 1: User Management with Phone Number as Primary Key
This creates/updates users and manages their preferences.
"""

from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional, Dict, Any


async def get_or_create_user(db: AsyncIOMotorDatabase, phone: str) -> Dict[str, Any]:
    """
    Get existing user by phone or create new one.
    Phone is the primary identifier for WhatsApp integration.
    
    Args:
        db: MongoDB database instance
        phone: User's phone number (format: +919876543210)
    
    Returns:
        User document with _id, phone, and preferences
    """
    # Check if user exists
    user = await db["users"].find_one({"phone": phone})
    
    if user:
        return user
    
    # Create new user with default preferences
    new_user = {
        "phone": phone,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
        
        # User preferences
        "preferences": {
            "personalityMode": "friendly",  # default mode
            "allowSavageMode": False,       # must opt-in
            "emojiLevel": "moderate",       # low, moderate, high
            "customName": None,             # How bot addresses user (e.g., "Boss", "Chief")
            "celebrateWins": True,          # Extra hype on profits
            "sympathizeOnLoss": True,       # Supportive on losses
            "milestoneNotifications": True, # Show conversation milestones
            "modeReminderShown": False      # One-time tutorial flag
        },
        
        # Conversation tracking
        "conversationStats": {
            "totalMessages": 0,
            "lastMessageAt": None,
            "lastSessionId": None
        },
        
        # Mode history (for analytics)
        "modeHistory": [
            {
                "mode": "friendly",
                "setAt": datetime.now(timezone.utc)
            }
        ]
    }
    
    result = await db["users"].insert_one(new_user)
    new_user["_id"] = result.inserted_id
    
    return new_user


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: ObjectId) -> Optional[Dict[str, Any]]:
    """Get user by ObjectId."""
    return await db["users"].find_one({"_id": user_id})


async def get_user_by_phone(db: AsyncIOMotorDatabase, phone: str) -> Optional[Dict[str, Any]]:
    """Get user by phone number."""
    return await db["users"].find_one({"phone": phone})


async def update_user_preference(
    db: AsyncIOMotorDatabase,
    phone: str,
    preference_key: str,
    preference_value: Any
) -> bool:
    """
    Update a specific user preference.
    
    Args:
        db: MongoDB database
        phone: User's phone number
        preference_key: Key in preferences dict (e.g., 'personalityMode')
        preference_value: New value
    
    Returns:
        True if updated successfully
    """
    result = await db["users"].update_one(
        {"phone": phone},
        {
            "$set": {
                f"preferences.{preference_key}": preference_value,
                "updatedAt": datetime.now(timezone.utc)
            }
        }
    )
    
    return result.modified_count > 0


async def increment_message_count(db: AsyncIOMotorDatabase, phone: str) -> int:
    """
    Increment user's message count and return new total.
    Used for milestone tracking.
    
    Returns:
        New total message count
    """
    result = await db["users"].find_one_and_update(
        {"phone": phone},
        {
            "$inc": {"conversationStats.totalMessages": 1},
            "$set": {
                "conversationStats.lastMessageAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
        },
        return_document=True  # Return updated document
    )
    
    if not result:
        return 0
    
    return result.get("conversationStats", {}).get("totalMessages", 0)


async def get_message_count(db: AsyncIOMotorDatabase, phone: str) -> int:
    """Get user's total message count."""
    user = await db["users"].find_one({"phone": phone})
    if not user:
        return 0
    return user.get("conversationStats", {}).get("totalMessages", 0)


