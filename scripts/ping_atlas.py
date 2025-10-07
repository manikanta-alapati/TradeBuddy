import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root
uri = os.getenv("MONGODB_URI")
print("Using URI:", (uri or "")[:40] + "...")  # don’t dump whole secret to console

client = MongoClient(uri, server_api=ServerApi("1"))
try:
  client.admin.command("ping")
  print("✅ Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
  print("❌ Mongo ping failed:", repr(e))
  raise
finally:
  client.close()
