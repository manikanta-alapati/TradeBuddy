# app/services/retrieval.py
from typing import List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.llm import embed_text
from app.services.vector import vector_search

# app/services/retrieval.py - UPDATE retrieve_context

async def retrieve_context(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    question: str,
    k: int = 5,
    kind: str = None
) -> List[Dict]:
    """
    Retrieve relevant chunks via vector search.
    
    UPDATED: Now searches across holdings AND positions.
    """
    from app.services.llm import embed_text  # ✅ Your actual module!
    
    # Generate query embedding
    query_embedding = embed_text(question)
    
    # Build search filter
    search_filter = {"userId": user_id}
    if kind:
        search_filter["kind"] = kind
    
    # Vector search
    results = await db["embeddings"].aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 3,
                "limit": k,
                "filter": search_filter
            }
        },
        {
            "$project": {
                "text": 1,
                "kind": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]).to_list(k)
    
    # Format for LLM
    chunks = []
    for r in results:
        # Include type info in chunk ID for clarity
        metadata = r.get("metadata", {})
        stock_type = metadata.get("type", "holding")  # "holding" or "position"
        symbol = metadata.get("symbol", "portfolio")
        
        chunks.append({
            "docId": f"{stock_type}:{symbol}",
            "chunk": r.get("text"),
            "score": r.get("score", 0.0),
            "metadata": metadata
        })
    
    return chunks