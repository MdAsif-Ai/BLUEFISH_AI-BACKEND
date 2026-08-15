"""
BlueFish AI - Digital Twin Routes (Government — 3D Simulator)
==============================================================
Page 3 of the Next.js Command Center. Marine ABM simulation.
All routes require `government` role.

  POST /api/v1/twin/run              → Enqueue simulation → returns task_id
  GET  /api/v1/twin/status/{task_id} → Poll for simulation result
  GET  /api/v1/twin/scenarios        → List policy scenarios
  GET  /api/v1/twin/history          → Past simulation results

Architecture:
  FastAPI returns task_id immediately.
  Celery worker runs the 90-day simulation in a background process.
  Result is stored in Redis (Celery result backend, DB 2).
  Frontend polls /status/{task_id} every 2 seconds.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from core.security import AuthenticatedUser, require_government

logger = logging.getLogger("bluefish.routes.twin")

router = APIRouter(
    prefix="/api/v1/twin",
    tags=["🌊 Digital Twin — 3D Simulator"],
    dependencies=[Depends(require_government)],
)


class SimulationRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=90)
    fleet_size: int = Field(default=100, ge=1, le=10000)
    policy_restrictions: Dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {'close_gulf_of_mannar': true, 'climate_sst_delta': 1.5}",
        examples=[{"close_gulf_of_mannar": True}],
    )
    initial_lat: float = Field(default=10.8)
    initial_lon: float = Field(default=79.8)


@router.post("/run", summary="Start Digital Twin ABM simulation (async via Celery)")
async def run_digital_twin(
    payload: SimulationRequest,
    user: AuthenticatedUser = Depends(require_government),
):
    """
    Enqueues the ABM simulation as a Celery task and returns a `task_id` immediately.

    The simulation may take 30–120 seconds depending on fleet_size and days.
    Poll `GET /api/v1/twin/status/{task_id}` to check progress.

    When status = 'SUCCESS', the response contains the full CesiumJS-compatible
    timeline JSON array for the 3D map animation.
    """
    try:
        from tasks import run_digital_twin_simulation

        task = run_digital_twin_simulation.apply_async(
            kwargs={
                "days": payload.days,
                "fleet_size": payload.fleet_size,
                "policy_restrictions": payload.policy_restrictions,
                "initial_lat": payload.initial_lat,
                "initial_lon": payload.initial_lon,
            },
            queue="simulation",
        )
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": f"Simulation enqueued. Poll /api/v1/twin/status/{task.id} for results.",
            "estimated_time_seconds": max(30, payload.days * payload.fleet_size // 1000),
            "params": {
                "days": payload.days,
                "fleet_size": payload.fleet_size,
                "policy_restrictions": payload.policy_restrictions,
            },
        }
    except Exception as e:
        logger.error(f"Failed to enqueue simulation: {e}", exc_info=True)
        # Celery unavailable — run synchronously as fallback
        logger.warning("Celery unavailable — running simulation synchronously (may be slow).")
        import asyncio
        from digital_twin.engine import run_simulation
        from core.model_loader import get_model_registry

        reg = get_model_registry()
        try:
            result = await run_simulation(
                days=payload.days,
                fleet_size=payload.fleet_size,
                policy_restrictions=payload.policy_restrictions,
                initial_lat=payload.initial_lat,
                initial_lon=payload.initial_lon,
                model_registry=reg,
            )
            return {
                "task_id": "sync_fallback",
                "status": "SUCCESS",
                "result": result,
                "note": "Celery unavailable — ran synchronously.",
            }
        except Exception as sync_e:
            raise HTTPException(500, detail=f"Simulation failed: {sync_e}")


@router.get("/status/{task_id}", summary="Poll digital twin simulation result")
async def get_simulation_status(
    task_id: str = Path(..., description="task_id returned by POST /twin/run"),
    user: AuthenticatedUser = Depends(require_government),
):
    """
    Checks the Celery result backend (Redis DB 2) for the simulation status.

    Possible status values:
      - PENDING  : Task not yet started (in queue)
      - STARTED  : Task is currently running in a worker
      - SUCCESS  : Complete — `result` contains the timeline JSON
      - FAILURE  : Task crashed — `error` contains the exception message
      - RETRY    : Task failed and is being retried

    The frontend should poll every 2-3 seconds and stop on SUCCESS or FAILURE.
    """
    if task_id == "sync_fallback":
        return {"task_id": task_id, "status": "SUCCESS", "message": "Synchronous run — result already returned."}

    try:
        from celery_worker import celery_app
        task_result = celery_app.AsyncResult(task_id)

        status = task_result.status          # PENDING, STARTED, SUCCESS, FAILURE, RETRY

        response: Dict[str, Any] = {"task_id": task_id, "status": status}

        if status == "SUCCESS":
            raw = task_result.result
            if isinstance(raw, dict) and raw.get("status") == "failed":
                response["status"] = "FAILURE"
                response["error"] = raw.get("error", "Unknown error in task")
            else:
                response["result"] = raw
        elif status == "FAILURE":
            response["error"] = str(task_result.result)
        elif status in ("PENDING", "STARTED", "RETRY"):
            response["message"] = "Simulation in progress. Poll again in 2-3 seconds."

        return response

    except Exception as e:
        logger.error(f"Status check failed for task {task_id}: {e}")
        raise HTTPException(500, detail=f"Failed to check task status: {e}")


@router.get("/scenarios", summary="List available policy scenarios")
async def list_scenarios(user: AuthenticatedUser = Depends(require_government)):
    """Pre-defined policy scenarios for the 3D simulator dropdown."""
    return {
        "scenarios": [
            {"id": "baseline", "name": "Baseline (No Restrictions)",
             "description": "Current fishing regulations — status quo.", "restrictions": {}},
            {"id": "gulf_closure", "name": "Close Gulf of Mannar MPA",
             "description": "Enforces the Gulf of Mannar Marine National Park boundary.",
             "restrictions": {"close_gulf_of_mannar": True}},
            {"id": "monsoon_ban", "name": "Monsoon Season Fishing Ban",
             "description": "61-day annual ban (April 15 – June 14).",
             "restrictions": {"monsoon_ban": True}},
            {"id": "climate_stress", "name": "Climate Stress (+1.5°C SST)",
             "description": "Simulates fish distribution shifts under a +1.5°C warming scenario.",
             "restrictions": {"climate_sst_delta": 1.5}},
            {"id": "combined_policy", "name": "MPA Closure + Climate Stress",
             "description": "Combined scenario: Gulf of Mannar closure under climate change.",
             "restrictions": {"close_gulf_of_mannar": True, "climate_sst_delta": 1.5}},
        ]
    }
