# scripts/create_indexes.py
"""
Create/ensure MongoDB indexes for the PortfolioRAG project.

Run from project root:
  - python scripts/create_indexes.py
  - OR: python -m scripts.create_indexes
"""

import asyncio
import os
import sys

# --- Make sure 'app' package is importable when running this file directly ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

from app.settings import settings
from app.mongo_collections import (
    USERS,
    CONNECTIONS,
    INSTRUMENTS,
    FUNDS,
    HOLDINGS,
    POSITIONS,
    ORDERS,
    TRADES,
    METRICS_DAILY,
    SYMBOL_SHEETS,
    MESSAGES,
    EMBEDDINGS,
)


async def ensure_indexes() -> None:
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]

    # USERS
    await db[USERS].create_index("phone", unique=True)

    # CONNECTIONS
    await db[CONNECTIONS].create_index([("userId", 1)])
    await db[CONNECTIONS].create_index([("expiresAt", -1)])

    # INSTRUMENTS (global, one per instrumentToken)
    await db[INSTRUMENTS].create_index("instrumentToken", unique=True)
    await db[INSTRUMENTS].create_index([("tradingsymbol", 1)])
    await db[INSTRUMENTS].create_index([("expiry", 1), ("strike", 1), ("instrumentType", 1)])

    # FUNDS (latest per segment)
    await db[FUNDS].create_index([("userId", 1), ("segment", 1)])

    # HOLDINGS (one per user + instrument)
    await db[HOLDINGS].create_index([("userId", 1), ("instrumentToken", 1)])

    # POSITIONS (separate docs for bucket: day | net)
    await db[POSITIONS].create_index([("userId", 1), ("instrumentToken", 1), ("bucket", 1)])

    # ORDERS (history; orderId is unique natural key)
    await db[ORDERS].create_index("orderId", unique=True)
    await db[ORDERS].create_index([("userId", 1), ("ts", -1)])

    # TRADES (history; tradeId is unique natural key)
    await db[TRADES].create_index("tradeId", unique=True)
    await db[TRADES].create_index([("userId", 1), ("ts", -1)])
    await db[TRADES].create_index([("orderId", 1)])

    # METRICS_DAILY (fast YTD/MTD aggregations)
    await db[METRICS_DAILY].create_index([("userId", 1), ("date", -1)])

    # SYMBOL_SHEETS (one per user + symbolMonth)
    await db[SYMBOL_SHEETS].create_index([("userId", 1), ("symbolMonth", 1)], unique=True)
    await db[SYMBOL_SHEETS].create_index([("instrumentToken", 1)])

    # MESSAGES (conversation log)
    await db[MESSAGES].create_index([("userId", 1), ("ts", -1)])

    # EMBEDDINGS (Atlas Vector Search will add a Search index via Atlas UI/API)
    await db[EMBEDDINGS].create_index([("userId", 1), ("kind", 1), ("docId", 1)])
    
    await db["messages"].create_index([("userId", 1), ("ts", -1), ("archived", 1)])
    await db["users"].create_index([("lastMilestoneNotified", 1)])

    client.close()


def main() -> None:
    try:
        asyncio.run(ensure_indexes())
        print("✅ Indexes ensured.")
    except Exception as e:
        print(f"❌ Failed to create indexes: {e}")
        raise


if __name__ == "__main__":
    main()
