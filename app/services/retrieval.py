# app/services/retrieval.py
from typing import List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.llm import embed_text
from app.services.vector import vector_search

async def retrieve_context(
    db: AsyncIOMotorDatabase,
    *,
    user_id: ObjectId,
    question: str,
    k: int = 5,
    kind: str = "portfolio_summary",
) -> List[Dict[str, Any]]:
    """
    1) embed question
    2) vector search within user's embeddings (by kind)
    Returns a list of {docId, chunk, score}.
    """
    qvec = embed_text(question)  # 1536-d
    hits = await vector_search(db, user_id=user_id, kind=kind, query_vector=qvec, k=k)
    return hits
