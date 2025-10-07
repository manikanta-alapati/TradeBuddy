# app/db.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.settings import settings

mongo_client: AsyncIOMotorClient | None = None

async def connect_to_mongo():
    """
    Create and return a DB handle (not just the client).
    """
    global mongo_client
    mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
    return mongo_client[settings.mongodb_db]

async def close_mongo_connection():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        mongo_client = None
