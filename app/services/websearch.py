# app/services/websearch.py - COMPLETE REPLACEMENT

from typing import List, Dict
from app.settings import settings

def web_search(query: str, k: int = 3) -> List[Dict]:
    """
    Returns compact list: [{docId, chunk, url, score}]
    """
    print(f"🌐 [WEB SEARCH] Starting search for: {query}")
    
    # ✅ CHECK 1: Tavily API key
    if not settings.tavily_api_key:
        print("❌ [WEB SEARCH] TAVILY_API_KEY not set in .env!")
        return [{
            "docId": "error",
            "chunk": "Web search is not configured. Administrator needs to add TAVILY_API_KEY to .env file.",
            "url": "",
            "score": 1.0
        }]
    
    try:
        # Import here to catch import errors
        from tavily import TavilyClient
        
        print(f"✅ [WEB SEARCH] Tavily client initialized")
        
        client = TavilyClient(api_key=settings.tavily_api_key)
        
        print(f"🔍 [WEB SEARCH] Calling Tavily API...")
        res = client.search(query=query, max_results=k)
        
        print(f"📦 [WEB SEARCH] Tavily response received")
        print(f"📊 [WEB SEARCH] Raw response keys: {res.keys() if res else 'None'}")
        
        results = res.get("results", [])
        print(f"✅ [WEB SEARCH] Found {len(results)} results")
        
        if not results:
            print(f"⚠️ [WEB SEARCH] Tavily returned 0 results for query: {query}")
            return [{
                "docId": "no-results",
                "chunk": f"No current information found for: {query}. This might be because the information doesn't exist yet or is not indexed.",
                "url": "",
                "score": 0.5
            }]
        
        out = []
        for i, r in enumerate(results):
            content = r.get("content", "")
            url = r.get("url", "")
            
            print(f"  Result {i+1}: {url} ({len(content)} chars)")
            
            out.append({
                "docId": f"web-{i+1}",
                "chunk": content[:700],
                "url": url,
                "score": 0.80 - i*0.02,
            })
        
        print(f"✅ [WEB SEARCH] Returning {len(out)} formatted results")
        return out
        
    except ImportError as e:
        print(f"❌ [WEB SEARCH] Tavily library not installed: {e}")
        return [{
            "docId": "error",
            "chunk": "Web search library not installed. Run: pip install tavily-python",
            "url": "",
            "score": 1.0
        }]
        
    except Exception as e:
        print(f"❌ [WEB SEARCH] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        return [{
            "docId": "error",
            "chunk": f"Web search failed with error: {str(e)}. Using knowledge from training data instead.",
            "url": "",
            "score": 1.0
        }]