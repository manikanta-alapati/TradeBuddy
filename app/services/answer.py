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
    persona: Persona = "professional",
    max_chars_context: int = 7000,
) -> str:
    """
    Build the full prompt and call the chat model. Truncates context if huge.
    """
    system = build_system_prompt(persona)
    ctx = render_context(context_chunks)
    if len(ctx) > max_chars_context:
        ctx = ctx[:max_chars_context] + "\n...[truncated]"

    user_message = f"""\
USER QUESTION:
{question}

CONTEXT:
{ctx}

INSTRUCTIONS:
- Answer using only the CONTEXT above. If insufficient, say what else you need.
- If numbers are asked, surface them clearly.
- If user asked "refresh", reply: "Refresh requested — please trigger the refresh job." and stop.
"""

    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
