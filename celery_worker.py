"""
BlueFish AI - Celery Application Initialization
================================================
Defines the Celery app instance used by all task modules.
This file is imported by both `tasks.py` and the FastAPI main.py.

Celery configuration:
  - Broker: Redis DB 1 (separate from the app cache on DB 0)
  - Result Backend: Redis DB 2
  - Celery Beat: Schedules Agent 2 (fleet polling) every 60s and
    Agent 3 (nightly data ingestion) at 02:00 IST daily.

Worker startup command:
    celery -A celery_worker worker --loglevel=info --concurrency=4

Beat scheduler startup command:
    celery -A celery_worker beat --loglevel=info

For Render.com: Run these as separate "Background Worker" services.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from celery import Celery
from celery.schedules import crontab

# ── Add MODELS directory to Python path ───────────────────────────────────────
_base = Path(__file__).parent
for _model_dir in (_base.parent / "MODELS").iterdir():
    if _model_dir.is_dir():
        sys.path.insert(0, str(_model_dir))
sys.path.insert(0, str(_base.parent / "AGENTS"))

logger = logging.getLogger("bluefish.celery")


def create_celery_app() -> Celery:
    """Factory that creates and configures the Celery application."""
    # We read settings directly from env vars here because Celery workers
    # don't run inside FastAPI's lifespan — they need their own config bootstrap.
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

    app = Celery(
        "bluefish",
        broker=broker_url,
        backend=backend_url,
        include=["tasks"],
    )

    app.conf.update(
        # ── Serialization ────────────────────────────────────────────────────
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # ── Result lifecycle ─────────────────────────────────────────────────
        result_expires=7200,            # Results live in Redis for 2 hours
        # ── Reliability ──────────────────────────────────────────────────────
        task_acks_late=True,            # Re-queue on worker crash
        worker_prefetch_multiplier=1,   # One task at a time per worker slot (AI tasks are heavy)
        # ── Timezone ─────────────────────────────────────────────────────────
        timezone="Asia/Kolkata",
        enable_utc=True,
        # ── Beat Schedule ────────────────────────────────────────────────────
        beat_schedule={
            # Agent 2: Fleet Command loop — every 60 seconds
            "fleet-command-every-60s": {
                "task": "tasks.run_fleet_command_cycle",
                "schedule": 60.0,
                "options": {"queue": "fleet"},
            },
            # Agent 3: Nightly data ingestion — 02:00 IST (20:30 UTC)
            "nightly-data-ingestion": {
                "task": "tasks.run_nightly_data_ingestion",
                "schedule": crontab(hour=20, minute=30),
                "options": {"queue": "ingestion"},
            },
            # Agent 4: Retraining check — daily at 03:00 IST (21:30 UTC)
            "daily-retraining-check": {
                "task": "tasks.check_and_trigger_retraining",
                "schedule": crontab(hour=21, minute=30),
                "options": {"queue": "training"},
            },
        },
        # ── Task Queues ───────────────────────────────────────────────────────
        # Different queues allow different concurrency settings per worker type.
        # Start workers with: celery -A celery_worker worker -Q fleet --concurrency=2
        task_routes={
            "tasks.run_fleet_command_cycle": {"queue": "fleet"},
            "tasks.run_nightly_data_ingestion": {"queue": "ingestion"},
            "tasks.check_and_trigger_retraining": {"queue": "training"},
            "tasks.run_digital_twin_simulation": {"queue": "simulation"},
        },
    )

    return app


# Singleton Celery app — imported by `tasks.py` and `main.py`
celery_app = create_celery_app()
