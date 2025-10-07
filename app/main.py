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


@app.get("/callback", response_class=HTMLResponse)
async def kite_callback(request: Request):
    """
    Zerodha redirects here after login:
    /callback?...&request_token=XXXX&action=login
    We also expect userId in the query: /callback?userId=<ObjectId>
    """
    q = request.query_params
    request_token = q.get("request_token")
    user_id_str = q.get("userId")

    if not request_token:
        raise HTTPException(status_code=400, detail="Missing request_token")
    if not user_id_str:
        raise HTTPException(status_code=400, detail="Missing userId in redirect URL")

    try:
        user_id = ObjectId(user_id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid userId")

    tokens = exchange_request_token_for_access_token(request_token)

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

    return HTMLResponse(
        """
        <html>
          <body style="font-family:system-ui;padding:24px">
            <h2>✅ Zerodha connected</h2>
            <p>Your access token was saved. You can close this tab.</p>
          </body>
        </html>
        """
    )
