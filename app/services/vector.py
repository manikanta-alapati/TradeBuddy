from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from datetime import datetime

EMBEDDINGS_COLL = "embeddings"

async def insert_embedding(
    db: AsyncIOMotorDatabase,
    *,
    user_id: ObjectId,
    kind: str,
    doc_id: str,
    vector: List[float],
    chunk: str,
    phone: str | None = None
) -> str:
    doc = {
        "userId": user_id,
        "phone": phone,
        "kind": kind,
        "docId": doc_id,
        "vector": vector,
        "chunk": chunk
    }
    res = await db[EMBEDDINGS_COLL].insert_one(doc)
    return str(res.inserted_id)
async def vector_search(
    db: AsyncIOMotorDatabase,
    *,
    user_id: ObjectId,
    kind: str,
    query_vector: List[float],
    k: int = 3
) -> List[Dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embeddings_vector",
                "queryVector": query_vector,
                "path": "vector",
                "numCandidates": 100,
                "limit": k,
                "filter": { "userId": user_id, "kind": kind }
            }
        },
        {
            "$project": {
                "_id": 0,                               # <-- exclude ObjectId
                "docId": 1,
                "chunk": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ]
    cursor = db["embeddings"].aggregate(pipeline)
    return [doc async for doc in cursor]


async def upsert_embedding(
    db: AsyncIOMotorDatabase,
    *,
    user_id: ObjectId,
    
    kind: str,
    doc_id: str,
    vector: List[float],
    chunk: str,
    metadata: Optional[Dict[str, Any]] = None,
    phone: str | None = None
) -> str:
    doc = {
        "userId": user_id,
        "phone": phone,
        "kind": kind,
        "docId": doc_id,
        "vector": vector,
        "chunk": chunk,
        "meta": metadata or {},
        "updatedAt": datetime.utcnow(),
    }
    res = await db[EMBEDDINGS_COLL].update_one(
        {"userId": user_id, "kind": kind, "docId": doc_id},
        {"$set": doc},
        upsert=True,
    )
    if res.upserted_id:
        return str(res.upserted_id)
    # fetch existing _id for convenience
    existing = await db[EMBEDDINGS_COLL].find_one({"userId": user_id, "kind": kind, "docId": doc_id}, {"_id": 1})
    return str(existing["_id"]) if existing else ""

