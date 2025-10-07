# app/services/prompt.py
from typing import Literal, List, Dict

Persona = Literal["professional", "friendly"]

SYSTEM_BASE = """\
You are TradeBot, a portfolio assistant. Be accurate and concise.
If you don't know, say so. Never make up numbers.
Use only the provided CONTEXT to answer user questions about their account.
If the user asks you to "refresh", do not answer; tell the caller to trigger a refresh job.
"""

PERSONA_STYLES = {
    "professional": "Tone: precise, clear, structured, no emojis.",
    "friendly": "Tone: warm, approachable, simple wording, a touch of encouragement.",
}

def build_system_prompt(persona: Persona = "professional") -> str:
    style = PERSONA_STYLES.get(persona, PERSONA_STYLES["professional"])
    return SYSTEM_BASE + "\n" + style

def render_context(chunks: List[Dict]) -> str:
    """
    chunks = [{docId, chunk, score}, ...]
    """
    lines = []
    for c in chunks:
        score = c.get("score")
        if isinstance(score, float):
            score = f"{score:.3f}"
        lines.append(f"- [{c.get('docId','?')}] (score {score}): {c.get('chunk','')}")
    return "\n".join(lines)
