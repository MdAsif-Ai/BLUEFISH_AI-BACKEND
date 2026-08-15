"""
BlueFish AI - MLOps Routes (Government — Pipelines Page)
=========================================================
Page 4 of the Next.js Command Center. Model and data pipeline management.
All routes require `government` role.

  GET  /api/v1/system/data-status     → Data ingestion pipeline status
  GET  /api/v1/system/models          → All model versions from model_registry
  POST /api/v1/system/promote-model   → Human-in-the-loop model promotion
  POST /api/v1/system/trigger-ingestion → Manually trigger nightly ingestion task
  GET  /api/v1/system/celery-health   → Celery worker health check
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.security import AuthenticatedUser, require_government

logger = logging.getLogger("bluefish.routes.mlops")

router = APIRouter(
    prefix="/api/v1/system",
    tags=["⚙️ MLOps — Pipelines"],
    dependencies=[Depends(require_government)],
)


class PromoteModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = Field(..., description="e.g. 'model1_pfz'")
    version_id: str = Field(..., description="version_id from model_registry table")
    approved_by: str = Field(..., description="Name/email of approving government official")
    approval_notes: Optional[str] = Field(default=None, description="Optional justification text")


class RejectModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    version_id: str
    rejected_by: str
    rejection_reason: str


@router.get("/data-status", summary="Data ingestion pipeline status")
async def get_data_status(user: AuthenticatedUser = Depends(require_government)):
    """
    Returns the status of the nightly data ingestion pipeline.
    Queries:
    1. The `data_audit_log` table in Supabase for recent run logs
    2. The Celery result backend for the most recent ingestion task result
    """
    from core.database import get_supabase
    from datetime import datetime, timezone, timedelta

    # ── Recent audit logs from Supabase ──────────────────────────────────────
    audit_logs = []
    try:
        db = get_supabase()
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        result = (
            db.table("data_audit_log")
            .select("run_timestamp, results")
            .gte("run_timestamp", week_ago)
            .order("run_timestamp", desc=True)
            .limit(10)
            .execute()
        )
        audit_logs = result.data or []
    except Exception as e:
        logger.warning(f"Could not fetch audit logs: {e}")

    # ── Redis cache status for today ─────────────────────────────────────────
    from core.redis import cache_get
    from datetime import date
    today = str(date.today())
    pfz_cached = await cache_get(f"model1:pfz:{today}") is not None
    ocean_cached = await cache_get(f"model2:ocean_features:{today}") is not None

    return {
        "today": today,
        "cache_status": {
            "pfz_map_cached": pfz_cached,
            "ocean_features_cached": ocean_cached,
        },
        "recent_ingestion_runs": audit_logs,
        "ingestion_schedule": "Daily at 02:00 IST (20:30 UTC) via Celery Beat",
    }


@router.get("/models", summary="All model versions from model_registry")
async def get_model_registry_entries(user: AuthenticatedUser = Depends(require_government)):
    """
    Returns all entries in the `model_registry` table, grouped by model_name.
    Includes staging, production, and archived versions for full history.
    Also returns the in-memory load status of each model from the current worker.
    """
    from core.database import get_supabase
    from core.model_loader import get_model_registry

    reg = get_model_registry()
    memory_status = {
        "model1_pfz": reg.model1 is not None,
        "model2_fronts": reg.model2 is not None,
        "model3_lstm": reg.model3 is not None,
        "model4_tft": reg.model4 is not None,
        "model5_density": reg.model5 is not None,
        "model6_anomaly": reg.model6_forest is not None,
        "model7_route": reg.model7 is not None,
        "model8_timewindow": reg.model8 is not None,
        "model9_kmeans": reg.model9_kmeans is not None,
        "model10_collision": reg.model10 is not None,
        "model11_climate": reg.model11_xgb is not None,
    }

    try:
        db = get_supabase()
        result = (
            db.table("model_registry")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        registry_rows = result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch model registry: {e}")
        registry_rows = []

    return {
        "in_memory_models": memory_status,
        "load_errors": reg.load_errors,
        "registry_entries": registry_rows,
        "total_entries": len(registry_rows),
    }


@router.post("/promote-model", summary="[Human-Gated] Promote staging model to production")
async def promote_model(
    payload: PromoteModelRequest,
    user: AuthenticatedUser = Depends(require_government),
):
    """
    The ONLY path for a staging model to reach production.
    Requires a government role JWT and explicit `approved_by` field.

    This endpoint:
    1. Verifies the model exists and is in 'staging' status
    2. Archives the current 'production' model (if any)
    3. Promotes the specified version to 'production'
    4. Logs the promotion with timestamp and approver identity

    This will NEVER be called automatically by any background task.
    """
    from core.database import get_supabase
    from agents.retraining_agent import build_retraining_agent

    db = get_supabase()
    agent = build_retraining_agent(db)

    try:
        agent.promote_to_production(
            model_name=payload.model_name,
            version_id=payload.version_id,
            approved_by=payload.approved_by,
        )
        return {
            "status": "promoted",
            "model_name": payload.model_name,
            "version_id": payload.version_id,
            "approved_by": payload.approved_by,
            "approval_notes": payload.approval_notes,
            "promoted_by_user_id": user.user_id,
            "message": f"✅ {payload.model_name} v{payload.version_id} is now in production.",
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.error(f"Model promotion failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Promotion failed: {e}")


@router.post("/reject-model", summary="Reject a staging model")
async def reject_model(
    payload: RejectModelRequest,
    user: AuthenticatedUser = Depends(require_government),
):
    """Archives a staging model with a rejection reason, preventing accidental promotion."""
    from core.database import get_supabase
    from datetime import datetime, timezone

    db = get_supabase()
    try:
        result = (
            db.table("model_registry")
            .update({
                "status": "archived",
                "metrics": {"rejection_reason": payload.rejection_reason,
                            "rejected_by": payload.rejected_by,
                            "rejected_at": datetime.now(timezone.utc).isoformat()},
            })
            .eq("model_name", payload.model_name)
            .eq("version_id", payload.version_id)
            .eq("status", "staging")
            .execute()
        )
        if not result.data:
            raise HTTPException(404, detail="Staging model version not found.")
        return {"status": "rejected", "model_name": payload.model_name,
                "version_id": payload.version_id, "reason": payload.rejection_reason}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Rejection failed: {e}")


@router.post("/trigger-ingestion", summary="Manually trigger the nightly data ingestion task")
async def trigger_ingestion(user: AuthenticatedUser = Depends(require_government)):
    """
    Manually enqueues the nightly data ingestion task without waiting for the Beat schedule.
    Useful for ad-hoc refreshes after satellite data becomes available.
    Returns a task_id to track progress.
    """
    try:
        from tasks import run_nightly_data_ingestion
        task = run_nightly_data_ingestion.apply_async(queue="ingestion")
        return {"task_id": task.id, "status": "PENDING",
                "message": f"Ingestion task enqueued. Poll /api/v1/twin/status/{task.id}"}
    except Exception as e:
        logger.error(f"Failed to enqueue ingestion task: {e}")
        raise HTTPException(500, detail=f"Failed to trigger ingestion: {e}")


@router.post("/trigger-retraining-check", summary="Manually trigger retraining eligibility check")
async def trigger_retraining_check(user: AuthenticatedUser = Depends(require_government)):
    """Manually triggers Agent 4 to check if enough feedback data exists for retraining."""
    try:
        from tasks import check_and_trigger_retraining
        task = check_and_trigger_retraining.apply_async(queue="training")
        return {"task_id": task.id, "status": "PENDING",
                "message": "Retraining check enqueued. NEVER auto-promotes — staging only."}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to trigger retraining check: {e}")


@router.get("/celery-health", summary="Celery worker health status")
async def get_celery_health(user: AuthenticatedUser = Depends(require_government)):
    """
    Pings the Celery control channel to discover active workers and their queues.
    Returns worker hostnames, queue assignments, and active task counts.
    """
    try:
        from celery_worker import celery_app
        inspector = celery_app.control.inspect(timeout=3.0)

        active = inspector.active() or {}
        registered = inspector.registered() or {}
        stats = inspector.stats() or {}

        workers = []
        for worker_name, tasks in active.items():
            workers.append({
                "name": worker_name,
                "active_tasks": len(tasks),
                "registered_tasks": len(registered.get(worker_name, [])),
            })

        return {
            "celery_status": "ok" if workers else "no_workers_online",
            "active_workers": len(workers),
            "workers": workers,
            "broker": "redis",
        }
    except Exception as e:
        logger.warning(f"Celery health check failed: {e}")
        return {"celery_status": "error", "detail": str(e), "active_workers": 0}
