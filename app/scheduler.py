# app/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
import datetime as dt
from datetime import timezone
import traceback

class Scheduler:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.run_incremental_sync = None
        self.build_embeddings_for_user = None

    def set_hooks(self, run_incremental_sync, build_embeddings_for_user):
        """Store sync functions to avoid circular imports."""
        self.run_incremental_sync = run_incremental_sync
        self.build_embeddings_for_user = build_embeddings_for_user

    def start(self):
        # Run every 15 minutes; keep only one instance if previous is still running
        self.scheduler.add_job(
            self.refresh_all_users_job,
            CronTrigger(minute="*/15"),
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.start()

    def shutdown(self):
        if getattr(self.scheduler, "running", False):
            self.scheduler.shutdown(wait=False)

    async def refresh_all_users_job(self):
        try:
            # Lazy imports to avoid circular references at import time
            from app.services.sync import list_active_users

            user_ids = await list_active_users(self.db)
            for uid in user_ids:
                try:
                    oid = ObjectId(uid) if isinstance(uid, str) else uid
                    await self.run_incremental_sync(self.db, oid)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()

    async def refresh_one_user(self, user_id, *, force_instruments=False, skip_embeddings=False):
        return await self.run_incremental_sync(
            self.db,
            user_id,
            force_instruments=force_instruments,
            skip_embeddings=skip_embeddings,
        )

    def enqueue_embeddings(self, user_id: ObjectId):
        self.scheduler.add_job(
            self.run_embeddings_job,
            trigger="date",                       # run once ASAP
            kwargs={"user_id": str(user_id)},
            id=f"embeddings-{str(user_id)}-{int(dt.datetime.now(timezone.utc).timestamp())}",
            max_instances=1,
            coalesce=True,
            replace_existing=False,
        )

    async def run_embeddings_job(self, user_id: str):
        uid = ObjectId(user_id) if isinstance(user_id, str) else user_id
        try:
            await self.build_embeddings_for_user(self.db, uid)
        except Exception as e:
            print(f"[Scheduler] Embeddings job failed for {uid}: {e}")