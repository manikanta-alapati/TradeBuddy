# app/services/retrieval.py - FIXED VERSION
from typing import List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.llm import embed_text
from app.services.vector import vector_search

async def retrieve_context(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    question: str,
    k: int = 5,
    kind: str = None
) -> List[Dict]:
    """
    Retrieve relevant chunks via vector search.
    FIXED: Correct field names and index name.
    """
    from app.services.llm import embed_text
    
    # Generate query embedding
    query_embedding = embed_text(question)
    
    # Build search filter
    search_filter = {"userId": user_id}
    if kind:
        search_filter["kind"] = kind
    
    try:
        # Try Atlas Vector Search first
        results = await db["embeddings"].aggregate([
            {
                "$vectorSearch": {
                    "index": "embeddings_vector",  # Your Atlas index name
                    "path": "vector",         # Changed from "embedding" to "vector"
                    "queryVector": query_embedding,
                    "numCandidates": k * 3,
                    "limit": k,
                    "filter": search_filter
                }
            },
            {
                "$project": {
                    "chunk": 1,
                    "kind": 1,
                    "docId": 1,
                    "meta": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]).to_list(k)
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        print(f"Falling back to text search...")
        
        # Fallback: Simple text search if vector search fails
        results = await db["embeddings"].find(
            {"userId": user_id, "$text": {"$search": question}},
            {"chunk": 1, "kind": 1, "docId": 1, "meta": 1}
        ).limit(k).to_list(k)
        
        # Add fake scores
        for r in results:
            r["score"] = 0.5
    
    # Format for LLM
    chunks = []
    for r in results:
        metadata = r.get("meta", {})
        chunks.append({
            "docId": r.get("docId", "unknown"),
            "chunk": r.get("chunk", ""),
            "score": r.get("score", 0.0),
            "metadata": metadata
        })
    
    # If no results, try getting ANY embeddings for this user
    if not chunks:
        print(f"⚠️ No vector search results for user {user_id}")
        
        # Get any portfolio data as fallback
        any_data = await db["embeddings"].find(
            {"userId": user_id},
            {"chunk": 1, "docId": 1}
        ).limit(3).to_list(3)
        
        if any_data:
            print(f"📊 Found {len(any_data)} fallback documents")
            for doc in any_data:
                chunks.append({
                    "docId": doc.get("docId", "fallback"),
                    "chunk": doc.get("chunk", ""),
                    "score": 0.3,
                    "metadata": {}
                })
        else:
            print(f"❌ No embeddings found for user {user_id}")
    
    return chunks