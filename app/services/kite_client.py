# app/services/kite_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import datetime as dt

from app.settings import settings

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None  # optional import so app can still boot without the package

class IKiteClient:
    def get_funds(self) -> Dict[str, Any]: ...
    def get_holdings(self) -> List[Dict[str, Any]]: ...
    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]: ...
    def get_orders(self) -> List[Dict[str, Any]]: ...
    def get_trades(self) -> List[Dict[str, Any]]: ...
    def get_instruments(self) -> List[Dict[str, Any]]: ...


class RealKiteClient(IKiteClient):
    def __init__(self, api_key: str, access_token: str):
        if KiteConnect is None:
            raise RuntimeError("kiteconnect not installed. Run: pip install kiteconnect")
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def profile(self):
        """Get user profile (for token validation)."""
        return self.kite.profile()

    def get_funds(self) -> Dict[str, Any]:
        return self.kite.margins()
    
    def get_holdings(self) -> List[Dict[str, Any]]:
        return self.kite.holdings()

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.kite.positions()

    def get_orders(self) -> List[Dict[str, Any]]:
        return self.kite.orders()

    def get_trades(self) -> List[Dict[str, Any]]:
        return self.kite.trades()

    def get_instruments(self) -> List[Dict[str, Any]]:
        return self.kite.instruments()
    
    

class StubKiteClient(IKiteClient):
    """Dev fallback so the app runs without a real token."""
    def get_funds(self) -> Dict[str, Any]:
        return {"equity": {"net": 100000, "available": 80000}}

    def get_holdings(self) -> List[Dict[str, Any]]:
        return [
            {"tradingsymbol": "TATAMOTORS", "quantity": 10, "average_price": 950.0, "last_price": 1020.0}
        ]

    def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"day": [], "net": []}

    def get_orders(self) -> List[Dict[str, Any]]:
        now = dt.datetime.utcnow()
        return [{"order_id": "ORD1", "tradingsymbol": "TATAMOTORS", "transaction_type": "BUY",
                 "quantity": 10, "average_price": 950.0, "status": "COMPLETE", "order_timestamp": now}]

    def get_trades(self) -> List[Dict[str, Any]]:
        now = dt.datetime.utcnow()
        return [{"trade_id": "TR1", "order_id": "ORD1", "tradingsymbol": "TATAMOTORS",
                 "quantity": 10, "price": 950.0, "trade_timestamp": now}]

    def get_instruments(self) -> List[Dict[str, Any]]:
        return [{"instrument_token": 123, "tradingsymbol": "TATAMOTORS", "exchange": "NSE", "segment": "EQ", "lot_size": 1}]

def build_kite_client(access_token: Optional[str], api_key: Optional[str] = None) -> IKiteClient:
    """
    If we have an access_token (real login done) -> RealKiteClient.
    Otherwise -> StubKiteClient (keeps the app running).
    """
    if access_token and (api_key or settings.kite_api_key):
        return RealKiteClient(api_key or settings.kite_api_key, access_token)  # type: ignore
    return StubKiteClient()

def exchange_request_token_for_access_token(request_token: str) -> Dict[str, str]:
    """
    Exchange Zerodha request_token for access_token using your app's api_key/secret from .env.
    Use this once per login, then store access_token in Mongo.
    """
    if KiteConnect is None:
        raise RuntimeError("kiteconnect not installed. Run: pip install kiteconnect")
    if not settings.kite_api_key or not settings.kite_api_secret:
        raise RuntimeError("KITE_API_KEY / KITE_API_SECRET missing in .env")

    kite = KiteConnect(api_key=settings.kite_api_key)
    data = kite.generate_session(request_token=request_token, api_secret=settings.kite_api_secret)
    # e.g. {'user_id': 'ABCD', 'access_token': '...', 'public_token': '...', ...}
    kite.set_access_token(data["access_token"])
    return {"access_token": data["access_token"], "public_token": data.get("public_token", "")}

# Add this to app/services/kite_client.py

async def validate_zerodha_token(db, user_id: ObjectId) -> tuple[bool, str]:
    """
    Validate if Zerodha token is still valid.
    Returns: (is_valid, error_message)
    """
    from datetime import datetime, timezone
    
    conn = await db["connections"].find_one({
        "userId": user_id,
        "provider": "zerodha",
        "enabled": True
    })
    
    if not conn:
        return False, "No Zerodha connection found"
    
    # Check if we marked it as expired
    if conn.get("tokenExpiredAt"):
        return False, "Token marked as expired"
    
    # Test the token
    kite = build_kite_client(conn.get("accessToken"), conn.get("apiKey"))
    
    try:
        profile = kite.profile()
        return True, "Token valid"
        
    except Exception as e:
        # Mark token as expired
        await db["connections"].update_one(
            {"_id": conn["_id"]},
            {
                "$set": {
                    "enabled": False,
                    "tokenExpiredAt": datetime.now(timezone.utc),
                    "lastError": str(e)
                }
            }
        )
        return False, "Token expired - please login again"