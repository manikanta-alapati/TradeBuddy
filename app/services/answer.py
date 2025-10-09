# app/services/answer.py
from typing import List, Dict
from openai import OpenAI
from app.settings import settings
from app.services.prompt import build_system_prompt, render_context, Persona

CHAT_MODEL = "gpt-4o-mini"  # fast & cost-effective; switchable

_client: OpenAI | None = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

def answer_with_context(
    question: str,
    context_chunks: List[Dict],
    persona: Persona = "friendly",
    max_chars_context: int = 7000,
    response_style: str = "whatsapp",
    conversation_history: str = None  # Changed to string (already formatted)
) -> str:
    """
    Build the full prompt and call the chat model.
    """
    print(f"[DEBUG answer_with_context] conversation_history received: {conversation_history is not None}")
    if conversation_history:
        print(f"[DEBUG] History length: {len(conversation_history)}")
        print(f"[DEBUG] First 300 chars: {conversation_history[:300]}")
    system = build_system_prompt(persona)
    ctx = render_context(context_chunks)
    if len(ctx) > max_chars_context:
        ctx = ctx[:max_chars_context] + "\n...[truncated]"

    # Build messages array
    messages = [{"role": "system", "content": system}]
    
    # Add conversation history if exists
    if conversation_history:
        messages.append({
            "role": "system",
            "content": f"[CONVERSATION HISTORY]\n{conversation_history}\n"
        })
    
    # Add portfolio context
    messages.append({
        "role": "system", 
        "content": f"[USER'S PORTFOLIO DATA]\n{ctx}\n"
    })
    
    # Length instruction based on response style
    length_instruction = ""
    if response_style == "whatsapp":
        length_instruction = """
RESPONSE LENGTH (CRITICAL FOR WHATSAPP):
- Keep responses SHORT (2-4 sentences for simple questions)
- Use line breaks for readability
- Brief answer + offer to explain more if needed
"""
    
    # Add current question
    messages.append({
        "role": "user",
        "content": f"{question}\n{length_instruction}"
    })

    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=300 if response_style == "whatsapp" else 1000
    )
    return resp.choices[0].message.content.strip()