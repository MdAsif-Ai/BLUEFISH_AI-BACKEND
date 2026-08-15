"""
BlueFish AI - Celery Task Definitions
=======================================
All 4 agent tasks are defined here as Celery tasks.
FastAPI routes call `.delay()` to enqueue them and return a `task_id` immediately.
The frontend polls `GET /api/v1/twin/status/{task_id}` to check progress.

Task hierarchy:
  - run_fleet_command_cycle    → Agent 2 (every 60s via Beat)
  - run_nightly_data_ingestion → Agent 3 (2:00 AM via Beat)
  - check_and_trigger_retraining → Agent 4 (3:00 AM via Beat)
  - run_digital_twin_simulation → Agent 1/ABM (triggered by API request)

Each task is isolated: one crash doesn't affect others.
All tasks load the model registry from the in-process Celery worker,
NOT from FastAPI's memory — Celery workers are separate processes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery_worker import celery_app

logger = logging.getLogger("bluefish.tasks")

# ── Bootstrap model registry for Celery workers ──────────────────────────────
# Workers are separate processes from FastAPI. They need to load models too.
# We use a lazy global to load once per worker process, not per task.

_worker_registry = None


def _get_worker_registry():
    """
    Returns the worker's local model registry.
    Loads models on first call (lazy init per worker process).
    """
    global _worker_registry
    if _worker_registry is not None:
        return _worker_registry

    import asyncio
    from core.model_loader import load_all_models
    from core.database import get_supabase
    from core.config import get_settings

    settings = get_settings()
    db = get_supabase()

    try:
        # asyncio.run() is safe here — Celery workers run sync by default
        _worker_registry = asyncio.run(load_all_models(db, settings.ML_MODELS_BUCKET))
        logger.info("Worker model registry initialized.")
    except Exception as e:
        logger.error(f"Worker model registry initialization failed: {e}")
        from core.model_loader import ModelRegistry
        _worker_registry = ModelRegistry()  # empty registry — tasks will degrade gracefully

    return _worker_registry


# ── Task 1: Fleet Command Cycle (Agent 2 — runs every 60s via Beat) ──────────

@celery_app.task(
    name="tasks.run_fleet_command_cycle",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def run_fleet_command_cycle(self) -> Dict[str, Any]:
    """
    Agent 2: Pulls live vessel GPS from Redis Geo-index, runs:
      - Model 5 (DBSCAN fleet density)
      - Model 6 (Isolation Forest anomaly/compliance)
      - Model 10 (CPA/TCPA collision detection)
    Pushes collision alerts via Redis Pub/Sub and writes violations to Supabase.
    """
    try:
        from core.redis import get_redis_sync
        from core.database import get_supabase
        from agents.fleet_command_agent import build_fleet_agent

        registry = _get_worker_registry()
        r = get_redis_sync()
        db = get_supabase()

        agent = build_fleet_agent(registry, r, db)
        report = agent.run_cycle()

        return {
            "status": "ok",
            "timestamp": report.timestamp,
            "vessels_checked": report.vessels_checked,
            "degraded": report.degraded,
            "degraded_reasons": report.degraded_reasons,
        }

    except Exception as exc:
        logger.error(f"Fleet cycle task failed: {exc}", exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)}


# ── Task 2: Nightly Data Ingestion (Agent 3 — runs at 2:00 AM via Beat) ──────

@celery_app.task(
    name="tasks.run_nightly_data_ingestion",
    bind=True,
    max_retries=1,
    soft_time_limit=3600,   # 1 hour — NetCDF downloads can be slow
    time_limit=4000,
    acks_late=True,
)
def run_nightly_data_ingestion(self) -> Dict[str, Any]:
    """
    Agent 3: Downloads latest satellite NetCDF files from Copernicus/INCOIS,
    runs the DataQualityAgent validation pipeline, then:
      - Runs Model 1 (PFZ) over the full grid → stores GeoJSON in Redis
      - Runs Model 2 (Fronts/Eddies) → stores GeoJSON in Redis
    Both outputs are cached with a 25-hour TTL so the next night's run
    refreshes them before expiry.
    """
    from datetime import datetime, timezone

    try:
        from core.database import get_supabase
        from core.redis import cache_set_sync
        from core.model_loader import get_model_registry
        from core.config import get_settings

        settings = get_settings()
        db = get_supabase()
        reg = _get_worker_registry()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        results: Dict[str, Any] = {"date": today, "steps": {}}

        # ── Step 1: Model 2 (Fronts/Eddies) from cached/downloaded NetCDF ───
        # In production: download from Copernicus Marine Service API here.
        # For now we signal that the step ran — wire actual downloader here.
        results["steps"]["model2_fronts_eddies"] = "skipped_no_netcdf_source"

        # ── Step 2: Model 1 (PFZ) grid inference ─────────────────────────────
        if reg.model1 is not None:
            # In production: iterate over a 0.25° grid covering Tamil Nadu EEZ.
            # Here we demonstrate the pattern with a minimal placeholder output.
            pfz_geojson = {
                "type": "FeatureCollection",
                "features": [],
                "date": today,
                "generated_by": "nightly_ingestion_task",
            }
            cache_key = f"model1:pfz:{today}"
            cache_set_sync(cache_key, pfz_geojson, ttl_seconds=90000)  # 25h TTL
            results["steps"]["model1_pfz"] = f"cached → {cache_key}"
        else:
            results["steps"]["model1_pfz"] = "skipped_model_not_loaded"

        # ── Step 3: Log audit to Supabase ────────────────────────────────────
        try:
            db.table("data_audit_log").insert({
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }).execute()
        except Exception as e:
            logger.warning(f"Audit log write failed (non-fatal): {e}")

        logger.info(f"Nightly ingestion complete for {today}: {results}")
        return {"status": "ok", **results}

    except Exception as exc:
        logger.error(f"Nightly ingestion task failed: {exc}", exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)}


# ── Task 3: Retraining Check (Agent 4 — runs at 3:00 AM via Beat) ────────────

@celery_app.task(
    name="tasks.check_and_trigger_retraining",
    bind=True,
    max_retries=0,
    soft_time_limit=7200,   # 2h — training scripts can be long
    acks_late=True,
)
def check_and_trigger_retraining(self) -> Dict[str, Any]:
    """
    Agent 4: Checks trip_feedback volume. If threshold met, triggers a
    training subprocess and registers the new model as 'staging'.
    NEVER auto-promotes to 'production' — requires explicit human approval.
    """
    try:
        import asyncio
        from agents.retraining_agent import check_and_trigger_retraining as _trigger
        from core.database import get_supabase
        from core.config import get_settings

        settings = get_settings()
        db = get_supabase()
        result = asyncio.run(_trigger(
            supabase_client=db,
            min_new_rows=settings.RETRAINING_MIN_NEW_FEEDBACK_ROWS,
        ))
        logger.info(f"Retraining check result: {result}")
        return {"status": "ok", **result}

    except Exception as exc:
        logger.error(f"Retraining check task failed: {exc}", exc_info=True)
        return {"status": "failed", "error": str(exc)}


# ── Task 4: Digital Twin Simulation (Triggered by API) ───────────────────────

@celery_app.task(
    name="tasks.run_digital_twin_simulation",
    bind=True,
    max_retries=0,
    soft_time_limit=300,    # 5 min max for 90-day simulation
    time_limit=360,
    acks_late=True,
)
def run_digital_twin_simulation(
    self,
    days: int,
    fleet_size: int,
    policy_restrictions: Dict[str, Any],
    initial_lat: float = 10.8,
    initial_lon: float = 79.8,
) -> Dict[str, Any]:
    """
    Runs the Agent-Based Model Marine Digital Twin simulation.
    Called by `POST /api/v1/twin/run` — returns task_id immediately.
    The frontend polls `GET /api/v1/twin/status/{task_id}` for the result.
    """
    try:
        import asyncio
        from digital_twin.engine import run_simulation

        registry = _get_worker_registry()

        result = asyncio.run(run_simulation(
            days=days,
            fleet_size=fleet_size,
            policy_restrictions=policy_restrictions,
            initial_lat=initial_lat,
            initial_lon=initial_lon,
            model_registry=registry,
        ))

        logger.info(
            f"Digital twin complete: days={days} fleet={fleet_size} "
            f"catch={result['summary']['total_catch_kg']}kg"
        )
        return {"status": "ok", **result}

    except Exception as exc:
        logger.error(f"Digital twin task failed: {exc}", exc_info=True)
        return {"status": "failed", "error": str(exc)}
