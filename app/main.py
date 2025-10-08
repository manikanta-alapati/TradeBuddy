# app/main.py
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import datetime as dt
from typing import List

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
async def ask(req: AskReq):
    if "refresh" in req.question.lower():
        return {"answer": "Refresh requested — please trigger the refresh job."}

    user_id = ObjectId(req.userId)

    # 1) retrieve internal context (vector search)
    chunks = await retrieve_context(app.state.mongodb, user_id=user_id, question=req.question, k=req.k, kind=req.kind)

    # 2) optional web search
    q_low = req.question.lower()
    wants_search = q_low.startswith("search:") or any(t in q_low for t in ["news", "market", "today", "headline", "analysis"])
    if wants_search:
        webq = req.question[7:].strip() if q_low.startswith("search:") else req.question
        web_chunks = web_search(webq, k=3)
        for wc in web_chunks:
            wc["docId"] = f"{wc['docId']} (web)"
        chunks = (chunks or []) + web_chunks

    # 3) answer
    answer = answer_with_context(req.question, chunks, persona=req.persona)

    # 4) minimal logging
    await app.state.mongodb["messages"].insert_many([
        {"userId": user_id, "role": "user", "text": req.question},
        {"userId": user_id, "role": "assistant", "text": answer},
    ])

    return {"answer": answer, "usedChunks": chunks}

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

@app.get("/callback", response_class=HTMLResponse)
async def kite_callback(request: Request):
    """
    Zerodha redirects here after login:
    /callback?status=success&request_token=XXXX&action=login&state=USER_ID
    """
    q = request.query_params
    request_token = q.get("request_token")
    status = q.get("status")
    user_id_str = q.get("state") or q.get("userId")  # Try both
    
    # Check status
    if status != "success":
        return HTMLResponse(
            """
            <html>
              <body style="font-family:system-ui;padding:24px">
                <h2>❌ Login Failed</h2>
                <p>Zerodha login was cancelled or failed.</p>
                <p>Please try again.</p>
              </body>
            </html>
            """
        )
    
    if not request_token:
        raise HTTPException(status_code=400, detail="Missing request_token")
    
    if not user_id_str:
        raise HTTPException(
            status_code=400, 
            detail="Missing userId. Please use the login URL from /zerodha/login-url endpoint"
        )
    
    try:
        user_id = ObjectId(user_id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid userId format")
    
    # Exchange request token for access token
    try:
        tokens = exchange_request_token_for_access_token(request_token)
    except Exception as e:
        return HTMLResponse(
            f"""
            <html>
              <body style="font-family:system-ui;padding:24px">
                <h2>❌ Token Exchange Failed</h2>
                <p>Error: {str(e)}</p>
                <p>Make sure your API key and secret are correct in .env file.</p>
              </body>
            </html>
            """
        )
    
    # Save to database
    db = request.app.state.mongodb
    await db["connections"].update_one(
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
        upsert=True,
    )
    
    # Get user phone for display
    user = await db["users"].find_one({"_id": user_id})
    phone = user.get("phone", "Unknown") if user else "Unknown"
    
    return HTMLResponse(
        f"""
        <html>
          <body style="font-family:system-ui;padding:24px;background:#f0f9ff">
            <div style="max-width:600px;margin:0 auto;background:white;padding:32px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.1)">
              <h2 style="color:#10b981;margin:0 0 16px 0">✅ Zerodha Connected!</h2>
              
              <div style="background:#ecfdf5;padding:16px;border-radius:8px;margin:16px 0">
                <p style="margin:0;color:#059669"><strong>User:</strong> {phone}</p>
                <p style="margin:8px 0 0 0;color:#059669;font-size:12px"><strong>User ID:</strong> {user_id_str}</p>
              </div>
              
              <p style="color:#374151">Your Zerodha account is now connected to TradeBuddy!</p>
              
              <div style="margin-top:24px;padding:16px;background:#fef3c7;border-radius:8px">
                <p style="margin:0;color:#92400e"><strong>⚠️ Important:</strong></p>
                <p style="margin:8px 0 0 0;color:#92400e;font-size:14px">
                  Zerodha access tokens expire daily. You'll need to re-login tomorrow.
                </p>
              </div>
              
              <div style="margin-top:24px">
                <p style="color:#6b7280;font-size:14px">Next steps:</p>
                <ol style="color:#6b7280;font-size:14px;padding-left:20px">
                  <li>Test your connection with: <code>/debug/ping-kite?userId={user_id_str}</code></li>
                  <li>Fetch your portfolio: <code>/debug/holdings?userId={user_id_str}</code></li>
                  <li>Trigger data sync: <code>/debug/refresh?userId={user_id_str}</code></li>
                </ol>
              </div>
              
              <p style="margin-top:24px;color:#9ca3af;font-size:12px">
                You can close this tab now.
              </p>
            </div>
          </body>
        </html>
        """
    )
    
    # Add to app/main.py

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