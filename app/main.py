# app/main.py
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import datetime as dt
from typing import List

from fastapi import Form
from fastapi.responses import Response
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from bson import ObjectId

from app.settings import settings
from app.db import connect_to_mongo, close_mongo_connection
from app.scheduler import Scheduler

# services
from app.services.vector import insert_embedding, vector_search
from app.services.llm import embed_text
from app.services.retrieval import retrieve_context
from app.services.answer import answer_with_context
from app.services.prompt import Persona
from app.services.websearch import web_search
from app.services.kite_client import (
    build_kite_client,
    exchange_request_token_for_access_token,
)
from app.services.mappers import map_holdings

from app.services.user_management import (
    get_or_create_user, 
    get_user_by_phone, 
    update_user_preference
)


# ---------------- helpers & models ----------------

def pad(vec, dim=1536):  # keep index at 1536; pad short demo vectors
    return vec + [0.0] * (dim - len(vec))

class DisconnectReq(BaseModel):
    userId: str

class KiteExchangeReq(BaseModel):
    userId: str
    requestToken: str  # from Zerodha redirect

class SaveAccessTokenReq(BaseModel):
    userId: str
    requestToken: str

class HealthResponse(BaseModel):
    status: str
    env: str
    version: str
    db_connected: bool

class EmbedTextReq(BaseModel):
    userId: str
    kind: str = "portfolio_summary"   # or "chat_summary", "symbol_month_summary"
    docId: str
    text: str

class AskReq(BaseModel):
    userId: str
    question: str
    persona: Persona = "professional"
    k: int = 5
    kind: str = "portfolio_summary"  # or "chat_summary", "symbol_month_summary"

class TradeIn(BaseModel):
    userId: str
    symbol: str
    qty: float
    price: float
    ts: datetime  # ISO-8601 string e.g. "2025-09-10T10:00:00Z"

class SeedTradesReq(BaseModel):
    trades: List[TradeIn]

class ConnectZerodhaReq(BaseModel):
    userId: str
    accessToken: str           # dev-only: paste manually (post-login)
    apiKey: str | None = None  # optional override
    apiSecret: str | None = None

# ---------------- lifespan ----------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongodb = await connect_to_mongo()
    app.state.temp_tokens = {} 
    # import here to avoid early import cycles
    from app.services.sync import run_incremental_sync, build_embeddings_for_user

    app.state.scheduler = Scheduler(app.state.mongodb)
    app.state.scheduler.set_hooks(
        run_incremental_sync=run_incremental_sync,
        build_embeddings_for_user=build_embeddings_for_user,
    )
    app.state.scheduler.start()

    try:
        yield
    finally:
        app.state.scheduler.shutdown()
        await close_mongo_connection()

app = FastAPI(
    title="PortfolioRAG",
    version="0.2.0",
    lifespan=lifespan,
)

# ---------------- health & root ----------------

@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    try:
        await app.state.mongodb.command("ping")
        db_ok = True
    except Exception as e:
        print("Mongo health ping failed:", repr(e))
        db_ok = False
    return HealthResponse(
        status="ok",
        env=settings.app_env,
        version="0.2.0",
        db_connected=db_ok,
    )

@app.get("/")
async def root():
    return {"message": "PortfolioRAG API is up. Try GET /healthz"}

# ---------------- embeddings debug ----------------

@app.post("/debug/seed-embeddings")
async def seed_embeddings():
    db = app.state.mongodb
    user_id = ObjectId()  # demo user

    v1 = pad([0.9, 0.1, 0.0])
    v2 = pad([0.85, 0.15, 0.0])
    v3 = pad([0.1, 0.9, 0.0])

    ids = []
    ids.append(await insert_embedding(db, user_id=user_id, kind="portfolio_summary",
                                      doc_id="DEMO-1", vector=v1, chunk="Close to v1"))
    ids.append(await insert_embedding(db, user_id=user_id, kind="portfolio_summary",
                                      doc_id="DEMO-2", vector=v2, chunk="Close to v1/v2"))
    ids.append(await insert_embedding(db, user_id=user_id, kind="portfolio_summary",
                                      doc_id="DEMO-3", vector=v3, chunk="Close to v3"))

    return {"userId": str(user_id), "inserted": ids}

@app.get("/debug/search-embeddings")
async def search_embeddings(userId: str, q: str = "v1", k: int = 2):
    db = app.state.mongodb
    user_id = ObjectId(userId)
    query_vec = pad([0.9, 0.1, 0.0]) if q == "v1" else pad([0.1, 0.9, 0.0])
    hits = await vector_search(db, user_id=user_id, kind="portfolio_summary", query_vector=query_vec, k=k)
    return {"matches": hits}

# ---------------- manual refresh ----------------

# app/main.py (your version already matches this, but double-check)
@app.post("/debug/refresh")
async def manual_refresh(
    userId: str,
    forceInstruments: bool = Query(False),
    skipEmbeddings: bool = Query(False),
):
    uid = ObjectId(userId)
    fast_result = await app.state.scheduler.refresh_one_user(
        uid,
        force_instruments=forceInstruments,
        skip_embeddings=True,  # <-- ensure fast path here
    )
    if not skipEmbeddings:
        app.state.scheduler.enqueue_embeddings(uid)
        fast_result["embeddings"] = "scheduled"
    return fast_result


# ---------------- kite quick checks ----------------

@app.get("/debug/ping-kite")
async def ping_kite(userId: str, request: Request):
    """
    Fail-fast validity check for Zerodha token. If expired, tells you to re-login.
    """
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        kite.profile()  # cheap call to validate token
    except Exception as e:
        return JSONResponse(
            {"error": "Zerodha access token invalid/expired. Please re-login.", "details": str(e)},
            status_code=401
        )
    return {"ok": True}

@app.get("/debug/holdings")
async def debug_holdings(userId: str, request: Request):
    """
    Small, fast data fetch to prove the connection and mapping.
    """
    db = request.app.state.mongodb    # ✅ FIX: read from request.app.state
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)

    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        raw = kite.get_holdings()
    except Exception as e:
        return JSONResponse(
            {"error": "Kite call failed (likely token expired). Please re-login.", "details": str(e)},
            status_code=401
        )

    holdings = map_holdings(raw)
    return {"count": len(holdings), "sample": holdings[:3]}

# ---------------- text -> embeddings, search ----------------

@app.post("/debug/embed-text")
async def debug_embed_text(req: EmbedTextReq):
    vec = embed_text(req.text)  # 1536-dim
    _id = await insert_embedding(
        app.state.mongodb,
        user_id=ObjectId(req.userId),
        kind=req.kind,
        doc_id=req.docId,
        vector=vec,
        chunk=req.text
    )
    return {"insertedId": _id}

@app.get("/debug/search-text")
async def debug_search_text(userId: str, query: str, k: int = 3, kind: str = "portfolio_summary"):
    qvec = embed_text(query)  # 1536-dim
    hits = await vector_search(
        app.state.mongodb,
        user_id=ObjectId(userId),
        kind=kind,
        query_vector=qvec,
        k=k
    )
    return {"matches": hits}

# ---------------- ask ----------------

@app.post("/ask")
async def ask(req: AskReq, responseStyle: str = "whatsapp"):
    """
    Ask a question with full conversation memory and milestone tracking.
    """
    from app.services.conversation import (
        get_conversation_context,
        handle_message_with_context,
        format_conversation_for_llm
    )
    
    user_id = ObjectId(req.userId)

    # 0) Get smart conversation context
    conversation_messages, context_type = await get_conversation_context(
        app.state.mongodb,
        user_id,
        max_tokens=16000
    )
    
    # Format for LLM
    conversation_history = format_conversation_for_llm(conversation_messages)

    # 1) Retrieve portfolio context (vector search)
    chunks = await retrieve_context(
        app.state.mongodb, 
        user_id=user_id, 
        question=req.question, 
        k=req.k, 
        kind=req.kind
    )

    # 2) Optional web search
    q_low = req.question.lower()
    wants_search = q_low.startswith("search:") or any(
        t in q_low for t in ["news", "market", "today", "headline", "analysis", "current", "latest"]
    )
    if wants_search:
        webq = req.question[7:].strip() if q_low.startswith("search:") else req.question
        web_chunks = web_search(webq, k=3)
        for wc in web_chunks:
            wc["docId"] = f"{wc['docId']} (web)"
        chunks = (chunks or []) + web_chunks

    # 3) Generate answer with conversation context
    answer = answer_with_context(
        req.question, 
        chunks, 
        persona=req.persona,
        response_style=responseStyle,
        conversation_history=conversation_history
    )

    # 4) Store message and check milestones
    result = await handle_message_with_context(
        app.state.mongodb,
        user_id,
        req.question,
        answer
    )
    
    response = {
        "answer": answer,
        "usedChunks": chunks,
        "contextType": context_type
    }
    
    # Add milestone notification if triggered
    if result.get("milestone"):
        response["milestone"] = result["milestone"]["message"]
    
    return response
# ---------------- seed trades ----------------

@app.post("/debug/seed-trades")
async def debug_seed_trades(req: SeedTradesReq):
    docs = []
    for t in req.trades:
        docs.append({
            "userId": ObjectId(t.userId),
            "symbol": t.symbol.upper(),
            "qty": t.qty,
            "price": t.price,
            "ts": t.ts,  # BSON datetime
        })
    if docs:
        await app.state.mongodb["trades"].insert_many(docs)
    return {"inserted": len(docs)}

# ---------------- zerodha connect / disconnect / callback ----------------

@app.post("/debug/connect-zerodha")
async def connect_zerodha(req: ConnectZerodhaReq):
    db = app.state.mongodb
    user_id = ObjectId(req.userId)
    doc = {
        "userId": user_id,
        "provider": "zerodha",
        "apiKey": req.apiKey or settings.kite_api_key,
        "apiSecret": req.apiSecret or settings.kite_api_secret,
        "accessToken": req.accessToken,  # encrypt at rest in prod
        "scopes": ["read"],
        "createdAt": dt.datetime.now(timezone.utc),
        "expiresAt": None,
        "enabled": True,
    }
    await db["connections"].update_one(
        {"userId": user_id, "provider": "zerodha"},
        {"$set": doc},
        upsert=True
    )
    return {"ok": True}

@app.post("/debug/disconnect-zerodha")
async def disconnect_zerodha(req: DisconnectReq):
    db = app.state.mongodb
    user_id = ObjectId(req.userId)
    await db["connections"].delete_one({"userId": user_id, "provider": "zerodha"})
    return {"ok": True}

@app.post("/debug/kite-exchange")
async def debug_kite_exchange(req: KiteExchangeReq):
    tokens = exchange_request_token_for_access_token(req.requestToken)  # {'access_token':..., 'public_token':...}
    user_id = ObjectId(req.userId)

    await app.state.mongodb["connections"].update_one(
        {"userId": user_id, "provider": "zerodha"},
        {"$set": {
            "userId": user_id,
            "provider": "zerodha",
            "apiKey": settings.kite_api_key,
            "apiSecret": settings.kite_api_secret,
            "accessToken": tokens["access_token"],
            "publicToken": tokens.get("public_token"),
            "scopes": ["read"],
            "enabled": True,
            "createdAt": dt.datetime.now(timezone.utc),
        }},
        upsert=True
    )
    return {"ok": True, "provider": "zerodha"}

# --- more fast Zerodha debug routes ---

@app.get("/debug/profile")
async def debug_profile(userId: str, request: Request):
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        p = kite.profile()
    except Exception as e:
        return JSONResponse({"error": "Token invalid/expired", "details": str(e)}, status_code=401)
    # return a tiny subset
    return {"userName": p.get("user_name"), "email": p.get("email"), "userShortname": p.get("user_shortname")}

@app.get("/debug/funds")
async def debug_funds(userId: str, request: Request):
    from app.services.mappers import map_funds
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        raw = kite.get_funds()
    except Exception as e:
        return JSONResponse({"error": "Token invalid/expired", "details": str(e)}, status_code=401)
    funds = map_funds(raw)
    return {"count": len(funds), "sample": funds[:3]}

@app.get("/debug/positions")
async def debug_positions(userId: str, request: Request):
    from app.services.mappers import map_positions
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        raw = kite.get_positions()
    except Exception as e:
        return JSONResponse({"error": "Token invalid/expired", "details": str(e)}, status_code=401)
    positions = map_positions(raw)
    return {"count": len(positions), "sample": positions[:3]}

@app.get("/debug/orders")
async def debug_orders(userId: str, request: Request):
    from app.services.mappers import map_orders
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        raw = kite.get_orders()
    except Exception as e:
        return JSONResponse({"error": "Token invalid/expired", "details": str(e)}, status_code=401)
    orders = map_orders(raw)
    return {"count": len(orders), "sample": orders[:3]}

@app.get("/debug/trades")
async def debug_trades(userId: str, request: Request):
    from app.services.mappers import map_trades
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    if not conn:
        return JSONResponse({"error": "No Zerodha connection for this userId"}, status_code=400)
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    try:
        raw = kite.get_trades()
    except Exception as e:
        return JSONResponse({"error": "Token invalid/expired", "details": str(e)}, status_code=401)
    trades = map_trades(raw)
    return {"count": len(trades), "sample": trades[:3]}



@app.post("/users/create")
async def create_user(phone: str):
    """Create or get user by phone number."""
    user = await get_or_create_user(app.state.mongodb, phone)
    
    return {
        "userId": str(user["_id"]),
        "phone": user["phone"],
        "preferences": user["preferences"],
        "createdAt": user["createdAt"].isoformat()
    }


@app.get("/users/by-phone")
async def get_user_by_phone_endpoint(phone: str):
    """Get user details by phone number."""
    user = await get_user_by_phone(app.state.mongodb, phone)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "userId": str(user["_id"]),
        "phone": user["phone"],
        "preferences": user["preferences"],
        "messageCount": user.get("conversationStats", {}).get("totalMessages", 0)
    }


@app.post("/users/preferences")
async def update_preference(phone: str, key: str, value: str):
    """Update user preference."""
    success = await update_user_preference(
        app.state.mongodb, 
        phone, 
        key, 
        value
    )
    
    return {"success": success, "preference": key, "value": value}
# Add this to app/main.py

@app.get("/zerodha/login-url")
async def get_zerodha_login_url(phone: str):
    """
    Generate Zerodha login URL for a user.
    
    Usage:
    GET /zerodha/login-url?phone=%2B919876543210
    
    Returns a URL to open in browser for Zerodha login.
    After login, user will be redirected to /callback with access token.
    """
    from app.services.user_management import get_or_create_user
    
    # Get or create user
    user = await get_or_create_user(app.state.mongodb, phone)
    user_id = str(user["_id"])
    
    # Build Zerodha login URL
    # Note: userId will be passed back via state parameter
    login_url = (
        f"https://kite.zerodha.com/connect/login?"
        f"v=3&"
        f"api_key={settings.kite_api_key}"
    )
    
    # We need to modify callback to accept userId
    # For now, return instructions
    return {
        "userId": user_id,
        "phone": phone,
        "loginUrl": login_url,
        "instructions": [
            "1. Click the loginUrl above",
            "2. Login with your Zerodha credentials",
            "3. After login, you'll be redirected to /callback",
            f"4. Make sure the redirect URL includes: ?userId={user_id}",
            "5. The callback will save your access token automatically"
        ],
        "fullLoginUrl": f"{login_url}&state={user_id}",
        "note": "Open 'fullLoginUrl' in your browser. The state parameter carries your userId."
    }


# Update the existing /callback endpoint to handle state parameter

@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback(request: Request):
    """
    OAuth callback from Zerodha.
    Stores request_token temporarily WITHOUT requiring state parameter.
    User will identify themselves by typing "done" in WhatsApp.
    """
    q = request.query_params
    request_token = q.get("request_token")
    status = q.get("status")
    
    if status != "success":
        return HTMLResponse("""
            <html><body style="font-family:system-ui;padding:24px;text-align:center">
                <h2>❌ Login Failed</h2>
                <p>Please go back to WhatsApp and type "login" to try again.</p>
            </body></html>
        """)
    
    if not request_token:
        raise HTTPException(status_code=400, detail="Missing request_token")
    
    # Store request_token temporarily WITHOUT phone number
    # User will claim it by typing "done" within 5 minutes
    request.app.state.temp_tokens[request_token] = {
        "phone": None,  # Will be claimed by whoever types "done" first
        "timestamp": dt.datetime.now(timezone.utc).timestamp(),
        "used": False
    }
    
    print(f"✅ Request token stored: {request_token[:20]}...")
    
    # Show success page
    return HTMLResponse("""
        <html>
          <body style="font-family:system-ui;padding:40px;text-align:center;background:#f0f9ff">
            <div style="max-width:500px;margin:0 auto;background:white;padding:32px;border-radius:12px">
              <h1 style="color:#10b981;font-size:48px;margin:0">✅</h1>
              <h2 style="color:#10b981;margin:16px 0">Login Successful!</h2>
              <p style="color:#374151;font-size:18px">Your Zerodha login was successful.</p>
              <div style="background:#ecfdf5;padding:16px;border-radius:8px;margin:24px 0">
                <p style="color:#059669;margin:0;font-weight:600">📱 Go back to WhatsApp</p>
                <p style="color:#059669;margin:8px 0 0 0;font-size:20px;font-weight:700">Type: done</p>
              </div>
              <p style="color:#9ca3af;font-size:14px;margin-top:24px">
                ⏱️ You have 5 minutes to complete this step
              </p>
              <p style="color:#9ca3af;font-size:12px;margin-top:8px">You can close this page now.</p>
            </div>
          </body>
        </html>
    """)

@app.get("/zerodha/connect", response_class=HTMLResponse)
async def zerodha_connect_page():
    """
    Show a page where users can paste their request token after Zerodha login.
    This solves the redirect URL state parameter issue.
    """
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Connect Zerodha - TradeBuddy</title>
        <style>
            body {
                font-family: system-ui, -apple-system, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background: #f0f9ff;
            }
            .container {
                background: white;
                padding: 32px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h2 {
                color: #1e40af;
                margin-top: 0;
            }
            .step {
                background: #eff6ff;
                padding: 16px;
                border-radius: 8px;
                margin: 16px 0;
                border-left: 4px solid #3b82f6;
            }
            input {
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                border: 2px solid #e5e7eb;
                border-radius: 6px;
                font-size: 14px;
                box-sizing: border-box;
            }
            button {
                background: #10b981;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                margin-top: 16px;
            }
            button:hover {
                background: #059669;
            }
            button:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }
            .success {
                background: #ecfdf5;
                color: #059669;
                padding: 16px;
                border-radius: 8px;
                margin-top: 16px;
                display: none;
            }
            .error {
                background: #fef2f2;
                color: #dc2626;
                padding: 16px;
                border-radius: 8px;
                margin-top: 16px;
                display: none;
            }
            code {
                background: #f3f4f6;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 13px;
            }
            .login-btn {
                background: #3b82f6;
                margin-bottom: 24px;
            }
            .login-btn:hover {
                background: #2563eb;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔗 Connect Zerodha Account</h2>
            
            <div class="step">
                <strong>Step 1:</strong> Enter your phone number (with country code)
            </div>
            
            <input type="text" id="phoneInput" placeholder="+919876543210" value="+91">
            
            <button class="login-btn" onclick="openZerodhaLogin()">
                Open Zerodha Login
            </button>
            
            <div class="step">
                <strong>Step 2:</strong> After logging in to Zerodha, you'll be redirected to a page with an error. 
                <br><br>
                <strong>Copy the <code>request_token</code></strong> from the URL and paste it below.
                <br><br>
                Example URL:<br>
                <code style="font-size:11px">http://localhost:8000/callback?request_token=<strong>ARzIw3...</strong></code>
            </div>
            
            <input type="text" id="requestToken" placeholder="Paste request_token here">
            
            <button onclick="connectZerodha()" id="connectBtn">
                ✅ Connect Zerodha
            </button>
            
            <div class="success" id="successMsg"></div>
            <div class="error" id="errorMsg"></div>
        </div>
        
        <script>
            async function openZerodhaLogin() {
                const phone = document.getElementById('phoneInput').value.trim();
                
                if (!phone || phone === '+91') {
                    alert('Please enter your phone number');
                    return;
                }
                
                try {
                    // Create user and get login URL
                    const response = await fetch(`/zerodha/login-url?phone=${encodeURIComponent(phone)}`);
                    const data = await response.json();
                    
                    // Store userId in localStorage for next step
                    localStorage.setItem('userId', data.userId);
                    
                    // Open Zerodha login in new tab
                    window.open(data.loginUrl, '_blank');
                    
                    alert('Zerodha login opened in new tab. After logging in, come back here and paste the request_token.');
                } catch (error) {
                    document.getElementById('errorMsg').textContent = 'Error: ' + error.message;
                    document.getElementById('errorMsg').style.display = 'block';
                }
            }
            
            async function connectZerodha() {
                const requestToken = document.getElementById('requestToken').value.trim();
                const userId = localStorage.getItem('userId');
                
                if (!requestToken) {
                    alert('Please paste the request_token from the redirect URL');
                    return;
                }
                
                if (!userId) {
                    alert('Please click "Open Zerodha Login" first');
                    return;
                }
                
                const btn = document.getElementById('connectBtn');
                btn.disabled = true;
                btn.textContent = 'Connecting...';
                
                try {
                    const response = await fetch(
                        `/debug/kite-exchange?userId=${userId}&requestToken=${requestToken}`,
                        { method: 'POST' }
                    );
                    
                    const data = await response.json();
                    
                    if (data.ok) {
                        document.getElementById('successMsg').innerHTML = `
                            <strong>✅ Success!</strong><br><br>
                            Your Zerodha account is now connected!<br><br>
                            <strong>User ID:</strong> ${userId}<br><br>
                            You can now close this page and start using TradeBuddy.
                        `;
                        document.getElementById('successMsg').style.display = 'block';
                        document.getElementById('errorMsg').style.display = 'none';
                        
                        // Clear the form
                        document.getElementById('requestToken').value = '';
                    } else {
                        throw new Error(data.error || 'Connection failed');
                    }
                } catch (error) {
                    document.getElementById('errorMsg').textContent = 'Error: ' + error.message;
                    document.getElementById('errorMsg').style.display = 'block';
                    document.getElementById('successMsg').style.display = 'none';
                } finally {
                    btn.disabled = false;
                    btn.textContent = '✅ Connect Zerodha';
                }
            }
        </script>
    </body>
    </html>
    """)
    
    
@app.post("/zerodha/exchange-token")
async def exchange_token_simple(userId: str, requestToken: str):
    """
    Simple endpoint to exchange Zerodha request token for access token.
    
    Usage:
    POST /zerodha/exchange-token?userId=XXX&requestToken=YYY
    """
    try:
        user_id = ObjectId(userId)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid userId")
    
    # Exchange token
    try:
        tokens = exchange_request_token_for_access_token(requestToken)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")
    
    # Save to database
    await app.state.mongodb["connections"].update_one(
        {"userId": user_id, "provider": "zerodha"},
        {"$set": {
            "userId": user_id,
            "provider": "zerodha",
            "apiKey": settings.kite_api_key,
            "apiSecret": settings.kite_api_secret,
            "accessToken": tokens["access_token"],
            "publicToken": tokens.get("public_token"),
            "scopes": ["read"],
            "enabled": True,
            "createdAt": dt.datetime.now(timezone.utc),
        }},
        upsert=True
    )
    
    return {
        "success": True,
        "userId": userId,
        "message": "Zerodha connected successfully!"
    }
    
@app.post("/debug/generate-embeddings")
    
async def generate_embeddings_manually(userId: str):
    """
    Manually trigger embeddings generation for a user.
    
    This will:
    1. Read portfolio data from MongoDB
    2. Generate summaries
    3. Create embeddings
    4. Store in embeddings collection
    """
    from app.services.sync import build_embeddings_for_user
    
    user_id = ObjectId(userId)
    result = await build_embeddings_for_user(app.state.mongodb, user_id)
    
    return result

#----------new session-----
@app.post("/conversation/new-session")
async def create_new_session(userId: str):
    """
    Start a fresh conversation session.
    Archives old messages but keeps them searchable.
    """
    from app.services.conversation import start_new_session
    
    user_id = ObjectId(userId)
    result = await start_new_session(app.state.mongodb, user_id)
    
    return result

@app.get("/debug/conversation-history")
async def debug_conversation_history(userId: str, limit: int = 20):
    """Debug: See recent conversation history"""
    user_id = ObjectId(userId)
    
    messages = await app.state.mongodb["messages"].find(
        {"userId": user_id}
    ).sort("ts", -1).limit(limit).to_list(None)
    
    result = []
    for msg in messages:
        result.append({
            "role": msg.get("role"),
            "text": msg.get("text"),
            "ts": msg.get("ts").isoformat() if msg.get("ts") else None
        })
    
    return {
        "total": len(result),
        "messages": result
    }
    

@app.get("/debug/conversation-context")
async def debug_conversation_context(userId: str):
    """Debug: See what context is being retrieved"""
    user_id = ObjectId(userId)
    
    # Get message count first
    count = await app.state.mongodb["messages"].count_documents({"userId": user_id})
    
    # Get recent messages directly
    messages = await app.state.mongodb["messages"].find(
        {"userId": user_id}
    ).sort("ts", -1).limit(50).to_list(None)
    
    # Format manually
    formatted_lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        text = msg.get("text", "")
        formatted_lines.append(f"{role}: {text}")
    
    formatted = "\n".join(formatted_lines)
    
    return {
        "messageCount": len(messages),
        "totalCount": count,
        "formattedHistory": formatted
    }



# Add these endpoints

# app/main.py - UPDATE THIS SECTION

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(...),
    MessageSid: str = Form(None),
    NumMedia: str = Form("0"),
    ProfileName: str = Form("User")
):
    """
    Twilio WhatsApp webhook endpoint.
    Receives incoming messages and responds.
    """
    from app.services.whatsapp import parse_incoming_whatsapp, create_twiml_response, sanitize_whatsapp_message
    from app.services.whatsapp_handler import handle_whatsapp_message
    
    print(f"\n{'='*60}")
    print(f"📱 [WEBHOOK] Incoming WhatsApp Message")
    print(f"{'='*60}")
    print(f"From: {From}")
    print(f"To: {To}")
    print(f"Body: {Body}")
    print(f"MessageSid: {MessageSid}")
    print(f"{'='*60}\n")
    
    # Parse incoming message
    form_data = {
        "From": From,
        "To": To,
        "Body": Body,
        "MessageSid": MessageSid,
        "NumMedia": NumMedia,
        "ProfileName": ProfileName
    }
    
    parsed = parse_incoming_whatsapp(form_data)
    
    print(f"📝 [WEBHOOK] Parsed phone: {parsed['from']}")
    print(f"📝 [WEBHOOK] Parsed message: {parsed['body']}")
    
    # Handle message
    try:
        response_text = await handle_whatsapp_message(
            app.state.mongodb,
            phone=parsed["from"],
            message=parsed["body"],
            profile_name=parsed["profile_name"]
        )
        
        print(f"\n✅ [WEBHOOK] Generated response:")
        print(f"Length: {len(response_text)} chars")
        print(f"Preview: {response_text[:200]}...")
        
        # ✅ NEW: Sanitize message to remove markdown formatting
        # This fixes Twilio error 63038 (invalid WhatsApp message format)
        clean_response = response_text
        
        print(f"\n🧹 [WEBHOOK] Sanitized response:")
        print(f"Original length: {len(response_text)} chars")
        print(f"Cleaned length:  {len(clean_response)} chars")
        print(f"Preview: {clean_response[:200]}...")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ [WEBHOOK] Error generating response: {e}")
        import traceback
        traceback.print_exc()
        
        clean_response = "Sorry, I encountered an error. Please try again."
    
    # Create TwiML response with sanitized message
    twiml = create_twiml_response(clean_response)
    
    print(f"📤 [WEBHOOK] Sending TwiML response:")
    print(f"{twiml[:500]}...")
    print(f"{'='*60}\n")
    
    return Response(content=twiml, media_type="application/xml")

@app.post("/whatsapp/send")
async def send_whatsapp_test(to: str, message: str):
    """
    Test endpoint to send WhatsApp messages.
    Usage: POST /whatsapp/send?to=%2B1234567890&message=Hello
    """
    from app.services.whatsapp import send_whatsapp_message
    
    result = send_whatsapp_message(to, message)
    return result


@app.get("/debug/zerodha-raw-data")
async def debug_zerodha_raw_data(userId: str, request: Request):
    """Get ALL raw data from Zerodha API to diagnose issues"""
    from app.services.kite_client import build_kite_client
    
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    
    if not conn:
        return {"error": "No connection found for this user"}
    
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    
    result = {
        "userId": userId,
        "connectionFound": True
    }
    
    # Test Holdings
    try:
        holdings = kite.get_holdings()
        result["holdings"] = holdings
        result["holdings_count"] = len(holdings)
    except Exception as e:
        result["holdings_error"] = str(e)
        result["holdings_count"] = 0
    
    # Test Trades (THIS IS THE KEY!)
    try:
        trades = kite.get_trades()
        result["trades"] = trades
        result["trades_count"] = len(trades)
    except Exception as e:
        result["trades_error"] = str(e)
        result["trades_count"] = 0
    
    # Test Orders
    try:
        orders = kite.get_orders()
        result["orders"] = orders
        result["orders_count"] = len(orders)
    except Exception as e:
        result["orders_error"] = str(e)
        result["orders_count"] = 0
    
    # Test Positions
    try:
        positions = kite.get_positions()
        result["positions_day"] = positions.get("day", [])
        result["positions_net"] = positions.get("net", [])
        result["positions_count"] = len(positions.get("day", [])) + len(positions.get("net", []))
    except Exception as e:
        result["positions_error"] = str(e)
        result["positions_count"] = 0
    
    # Test Funds
    try:
        funds = kite.get_funds()
        result["funds"] = funds
    except Exception as e:
        result["funds_error"] = str(e)
    
    return result


@app.get("/debug/zerodha-orders-detail")
async def debug_zerodha_orders_detail(userId: str, request: Request):
    """Get detailed order history from Zerodha"""
    from app.services.kite_client import build_kite_client
    
    db = request.app.state.mongodb
    conn = await db["connections"].find_one({"userId": ObjectId(userId), "provider": "zerodha"})
    
    if not conn:
        return {"error": "No connection"}
    
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    
    try:
        all_orders = kite.get_orders()
        
        # Separate by status
        complete_orders = [o for o in all_orders if o.get('status') == 'COMPLETE']
        other_orders = [o for o in all_orders if o.get('status') != 'COMPLETE']
        
        # Build trade history from complete orders
        trade_history = []
        for order in complete_orders:
            trade_history.append({
                "date": str(order.get('order_timestamp')),
                "symbol": order.get('tradingsymbol'),
                "side": order.get('transaction_type'),
                "qty": order.get('filled_quantity'),
                "price": order.get('average_price'),
                "status": order.get('status')
            })
        
        return {
            "total_orders": len(all_orders),
            "completed_orders": len(complete_orders),
            "other_orders": len(other_orders),
            "trade_history": trade_history,
            "raw_sample": complete_orders[:3] if complete_orders else []
        }
    except Exception as e:
        return {"error": str(e)}
    
    
    # app/main.py - ADD THIS

@app.get("/debug/test-tavily")
async def test_tavily_direct():
    """Test Tavily API directly"""
    from app.services.websearch import web_search
    
    test_query = "upcoming IPOs India October 2025"
    
    print(f"\n{'='*60}")
    print(f"TESTING TAVILY API")
    print(f"Query: {test_query}")
    print(f"{'='*60}\n")
    
    try:
        results = web_search(test_query, k=3)
        
        return {
            "status": "success",
            "query": test_query,
            "results_count": len(results),
            "tavily_api_key_set": bool(settings.tavily_api_key),
            "tavily_api_key_prefix": settings.tavily_api_key[:10] + "..." if settings.tavily_api_key else "NOT SET",
            "results": results
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tavily_api_key_set": bool(settings.tavily_api_key)
        }
        
        
# app/main.py - ADD THIS DEBUG ENDPOINT

@app.post("/debug/send-test-message")
async def send_test_whatsapp(to: str, message: str = "Test from TradeBuddy!"):
    """
    Test sending WhatsApp message directly.
    Usage: POST /debug/send-test-message?to=%2B919876543210&message=Hello
    """
    from app.services.whatsapp import send_whatsapp_message
    
    print(f"\n{'='*60}")
    print(f"📤 [TEST] Sending test WhatsApp message")
    print(f"To: {to}")
    print(f"Message: {message}")
    print(f"{'='*60}\n")
    
    result = send_whatsapp_message(to, message)
    
    print(f"\n{'='*60}")
    print(f"📊 [TEST] Send result: {result}")
    print(f"{'='*60}\n")
    
    return result


# app/main.py - ADD THIS

@app.get("/debug/twilio-status")
async def check_twilio_status():
    """Check if messages are being delivered"""
    from twilio.rest import Client
    
    try:
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        
        # Get last 5 messages
        messages = client.messages.list(limit=5)
        
        result = []
        for msg in messages:
            result.append({
                "from": msg.from_,
                "to": msg.to,
                "body": msg.body[:100],
                "status": msg.status,  # ← IMPORTANT!
                "error_code": msg.error_code,
                "error_message": msg.error_message,
                "date": str(msg.date_created)
            })
        
        return {
            "messages_sent": len(result),
            "messages": result
        }
        
    except Exception as e:
        return {"error": str(e)}
    
    
    
    
    
# app/main.py - ADD THIS DEBUG ENDPOINT

# app/main.py - UPDATE the test endpoint

@app.get("/debug/test-emoji-sanitizer")
async def test_emoji_sanitizer():
    """Test emoji sanitization with real portfolio message"""
    from app.services.whatsapp import sanitize_whatsapp_message
    
    test_cases = [
        {
            "name": "Portfolio with problematic emojis",
            "input": "📊 **Your Portfolio**\n\n💼 Holdings: 7 stocks\n💰 Total Value: ₹79,041.30\n🟢 P&L: ₹9,856.75 (+14.25%)\n\n💡 Ask me about any stock!"
        },
        {
            "name": "Message with smart quotes (ACTUAL FAILURE)",
            "input": "Hiiiii! What's on your mind today? If you have any investment questions or need advice, I'm all ears!"
        },
        {
            "name": "Message with em dashes",
            "input": "Your portfolio — looking great — is up 14% today!"
        },
        {
            "name": "Message with ellipsis",
            "input": "Hmm… let me think about that…"
        }
    ]
    
    results = []
    for test in test_cases:
        original = test["input"]
        sanitized = sanitize_whatsapp_message(original)
        
        # Check for problematic characters
        has_smart_quote = ''' in original or ''' in original
        has_em_dash = '—' in original
        has_ellipsis = '…' in original
        
        results.append({
            "name": test["name"],
            "original": original,
            "sanitized": sanitized,
            "had_smart_quotes": has_smart_quote,
            "had_em_dash": has_em_dash,
            "had_ellipsis": has_ellipsis,
            "changed": original != sanitized
        })
    
    return {
        "total_tests": len(results),
        "results": results
    }