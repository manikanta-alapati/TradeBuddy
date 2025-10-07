# app/models/mongo_collections.py

USERS = "users"
CONNECTIONS = "connections"
INSTRUMENTS = "instruments"
FUNDS = "funds"
HOLDINGS = "holdings"
POSITIONS = "positions"
ORDERS = "orders"
TRADES = "trades"
METRICS_DAILY = "metrics_daily"
SYMBOL_SHEETS = "symbol_sheets"
MESSAGES = "messages"
EMBEDDINGS = "embeddings"

# Notes:
# - All docs should include userId (ObjectId) except instruments (global).
# - ORDERS use Zerodha orderId as a natural unique key; TRADES have tradeId.
