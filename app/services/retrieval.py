# app/services/retrieval.py - FIXED with correct index name

from typing import List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

async def retrieve_context(
    db: AsyncIOMotorDatabase,
    user_id: ObjectId,
    question: str,
    k: int = 5,
    kind: str = None
) -> List[Dict]:
    """
    Retrieve relevant chunks via vector search.
    FIXED: Using correct Atlas index name 'embeddings_vector'
    """
    from app.services.llm import embed_text
    
    # Generate query embedding
    query_embedding = embed_text(question)
    
    # Build search filter
    search_filter = {"userId": user_id}
    if kind:
        search_filter["kind"] = kind
    
    try:
        # MongoDB Atlas Vector Search with CORRECT index name
        results = await db["embeddings"].aggregate([
            {
                "$vectorSearch": {
                    "index": "embeddings_vector",  # ← FIXED! Using your actual index name
                    "path": "vector",               # ← Matches your Atlas config
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k,
                    "filter": search_filter
                }
            },
            {
                "$project": {
                    "chunk": 1,
                    "text": 1,
                    "kind": 1,
                    "docId": 1,
                    "meta": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]).to_list(k)
        
        print(f"✅ Vector search returned {len(results)} results")
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        
        # Fallback: Just get most recent embeddings
        results = await db["embeddings"].find(
            {"userId": user_id}
        ).sort("_id", -1).limit(k).to_list(k)
        
        for i, r in enumerate(results):
            r["score"] = 0.5 - (i * 0.05)
            
        print(f"📊 Using fallback: {len(results)} documents")
    
    # Format for LLM
    chunks = []
    for r in results:
        # Get text content (might be stored as 'chunk' or 'text')
        text_content = r.get("chunk") or r.get("text") or ""
        metadata = r.get("metadata") or r.get("meta", {})
        
        chunks.append({
            "docId": r.get("docId", "unknown"),
            "chunk": text_content,
            "score": r.get("score", 0.0),
            "metadata": metadata
        })
    
    return chunks