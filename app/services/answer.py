# app/services/answer.py

from typing import List, Dict
import json
from openai import OpenAI
from app.settings import settings
from app.services.prompt import build_system_prompt, render_context, Persona
from app.services.websearch import web_search

CHAT_MODEL = "gpt-4o-mini"

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing in .env")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


# ✅ Define tools (functions) the LLM can call
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": """Search the web for current, real-time information. Use this tool when:
            - User asks about IPOs, upcoming listings, public offerings
            - Query mentions "upcoming", "latest", "current", "today", "recent"
            - User asks about stock prices, market news, or financial events
            - Information requires real-time data beyond your knowledge cutoff
            - User explicitly asks you to search or get latest information
            
            DO NOT use this for:
            - User's own portfolio data (already provided in context)
            - General financial knowledge you already have
            - Historical facts that don't change""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query optimized for web search (e.g., 'upcoming IPOs India October 2025', 'latest NIFTY 50 price')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def answer_with_context(
    question: str,
    context_chunks: List[Dict],
    persona: Persona = "friendly",
    max_chars_context: int = 7000,
    response_style: str = "whatsapp",
    conversation_history: List[Dict] = None
) -> str:
    """
    Build the full prompt and call the chat model with tool support.
    LLM can now decide when to search the web intelligently.
    
    Args:
        question: User's current question
        context_chunks: Retrieved portfolio/trade data chunks
        persona: Bot personality mode
        max_chars_context: Max characters for portfolio context
        response_style: Response formatting style
        conversation_history: List of previous messages
    
    Returns:
        AI-generated response text
    """
    # Build system prompt and portfolio context
    system = build_system_prompt(persona)
    ctx = render_context(context_chunks)
    if len(ctx) > max_chars_context:
        ctx = ctx[:max_chars_context] + "\n...[truncated]"

    # Build messages array
    messages = [{"role": "system", "content": system}]
    
    # Add portfolio context as system message
    if ctx:
        messages.append({
            "role": "system", 
            "content": f"[USER'S PORTFOLIO DATA]\n{ctx}\n"
        })
    
    # Add conversation history as proper message objects
    if conversation_history:
        recent_history = conversation_history[-10:]  # Last 10 messages
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["text"]
            })
    
    # Smart length instruction based on query type
    length_instruction = ""
    if response_style == "whatsapp":
        detail_keywords = [
            "compare", "explain", "analyze", "breakdown", "detailed",
            "all", "everything", "comprehensive", "pros and cons"
        ]
        needs_detail = any(kw in question.lower() for kw in detail_keywords)
        
        if needs_detail:
            length_instruction = "\n\n[User wants DETAILED information. Provide complete answer.]"
        else:
            length_instruction = "\n\n[Keep response SHORT and conversational - 2-4 sentences for simple questions.]"
    
    # Add current question
    messages.append({
        "role": "user",
        "content": f"{question}{length_instruction}"
    })

    client = get_client()
    
    # ============================================
    # FIRST API CALL: Let LLM decide if it needs tools
    # ============================================
    
    print(f"🤖 [ANSWER] Making initial API call...")
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,  # Lower temperature for consistency
            max_tokens=2000,  # High enough for detailed answers
            tools=TOOLS,  # Give LLM access to web_search
            tool_choice="auto"  # Let LLM decide when to use tools
        )
    except Exception as e:
        print(f"❌ [ANSWER] OpenAI API error: {e}")
        return f"I encountered an error processing your request. Please try again. Error: {str(e)}"
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # ============================================
    # CHECK: Did LLM want to use tools?
    # ============================================
    
    if tool_calls:
        print(f"🔧 [ANSWER] LLM wants to use {len(tool_calls)} tool(s)")
        
        # Add LLM's tool decision to conversation
        messages.append(response_message)
        
        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            
            # Parse function arguments safely
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"❌ [ANSWER] Failed to parse tool arguments: {e}")
                print(f"    Raw arguments: {tool_call.function.arguments}")
                # Fallback to using the original question
                function_args = {"query": question}
            
            print(f"📞 [ANSWER] Calling {function_name}")
            print(f"    Args: {function_args}")
            
            if function_name == "web_search":
                # Execute web search
                search_query = function_args.get("query", question)
                print(f"🌐 [ANSWER] Executing web search: '{search_query}'")
                
                try:
                    # Call web search function
                    search_results = web_search(search_query, k=5)
                    
                    print(f"📊 [ANSWER] Web search returned {len(search_results)} results")
                    
                    # Format results for LLM
                    if search_results:
                        # Check if results contain errors
                        first_result = search_results[0]
                        
                        if first_result.get("docId") == "error":
                            # Web search failed
                            results_text = f"⚠️ Web Search Error: {first_result.get('chunk', 'Unknown error')}\n\nPlease answer using your training data and inform the user you don't have real-time information."
                            print(f"⚠️ [ANSWER] Web search returned error")
                            
                        elif first_result.get("docId") == "no-results":
                            # No results found
                            results_text = "⚠️ No current web results found.\n\nInform the user that you don't have real-time data for this query. Suggest they check official sources like NSE, BSE, or MoneyControl for current information."
                            print(f"⚠️ [ANSWER] Web search found no results")
                            
                        else:
                            # Valid results - format them
                            results_text = "📰 Web Search Results (Current Information):\n\n"
                            for i, result in enumerate(search_results, 1):
                                chunk = result.get('chunk', '')
                                url = result.get('url', 'N/A')
                                
                                results_text += f"{i}. {chunk}\n"
                                if url and url != 'N/A':
                                    results_text += f"   📎 Source: {url}\n"
                                results_text += "\n"
                            
                            print(f"✅ [ANSWER] Formatted {len(search_results)} valid results")
                    else:
                        # Empty results array
                        results_text = "⚠️ Web search returned no results.\n\nInform the user you don't have current data for this query."
                        print(f"⚠️ [ANSWER] Empty results array")
                    
                    print(f"📄 [ANSWER] Results text length: {len(results_text)} characters")
                    
                except Exception as e:
                    # Web search threw an exception
                    print(f"❌ [ANSWER] Web search exception: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    results_text = f"⚠️ Web search failed: {str(e)}\n\nPlease answer using your knowledge and inform the user you don't have real-time information for this query."
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": results_text
                })
        
        # ============================================
        # SECOND API CALL: Get final answer with tool results
        # ============================================
        
        print(f"🤖 [ANSWER] Making final API call with tool results...")
        
        try:
            final_response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            final_answer = final_response.choices[0].message.content.strip()
            print(f"✅ [ANSWER] Final answer generated ({len(final_answer)} chars)")
            
            return final_answer
            
        except Exception as e:
            print(f"❌ [ANSWER] Final API call error: {e}")
            return f"I found some information but encountered an error generating the response. Please try again."
    
    else:
        # ============================================
        # NO TOOLS NEEDED - Direct answer
        # ============================================
        
        print(f"💬 [ANSWER] LLM answered directly (no tools needed)")
        
        if response_message.content:
            return response_message.content.strip()
        else:
            print(f"⚠️ [ANSWER] Empty response from LLM")
            return "I'm having trouble generating a response. Please try rephrasing your question."