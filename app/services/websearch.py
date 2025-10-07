# app/services/websearch.py
from typing import List, Dict
from tavily import TavilyClient
from app.settings import settings

def web_search(query: str, k: int = 3) -> List[Dict]:
    """
    Returns compact list: [{docId, chunk, url, score}]
    We’ll treat these like vector chunks in the context.
    """
    if not settings.tavily_api_key:
        return []
    client = TavilyClient(api_key=settings.tavily_api_key)
    res = client.search(query=query, max_results=k)
    out = []
    for i, r in enumerate(res.get("results", [])):
        out.append({
            "docId": f"web-{i+1}",
            "chunk": r.get("content", "")[:700],  # keep it short
            "url": r.get("url"),
            "score": 0.80 - i*0.02,              # simple descending score
        })
    return out
