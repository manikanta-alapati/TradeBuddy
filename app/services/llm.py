# app/services/llm.py
from typing import List
from openai import OpenAI
from app.settings import settings

EMBED_MODEL = "text-embedding-3-small"   # 1536 dims

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

def embed_text(text: str) -> List[float]:
    """
    Returns a 1536-dim embedding for the given text using OpenAI.
    """
    client = get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    # OpenAI returns a single embedding for single input
    return resp.data[0].embedding
