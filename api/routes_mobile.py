"""
BlueFish AI - Mobile Routes (Fisherman Flutter App)
====================================================
All routes here require a valid JWT but only `fisherman` role.
Government users CAN access these (they also have access), but these
routes are optimized for the mobile use case.

  GET  /api/v1/mobile/map?date=YYYY-MM-DD  → Combined PFZ + ocean features GeoJSON
  POST /api/v1/mobile/route                 → Route optimization + time windows
  GET  /api/v1/mobile/recommendation        → Full advisory (graceful degradation)
  POST /api/v1/mobile/telemetry             → Submit vessel GPS position
  GET  /api/v1/mobile/alerts                → Safety alerts for this vessel
"""

from __future__ import annotations

import logging
from datetime import date as dt_date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.security import AuthenticatedUser, get_current_user
from core.redis import cache_get
from core.model_loader import get_model_registry

logger = logging.getLogger("bluefish.routes.mobile")

router = APIRouter(
    prefix="/api/v1/mobile",
    tags=["📱 Mobile — Fisherman App"],
    dependencies=[Depends(get_current_user)],  # All mobile routes require auth
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    start_lat: float = Field(..., ge=-90.0, le=90.0)
    start_lon: float = Field(..., ge=-180.0, le=180.0)
    target_lat: float = Field(..., ge=-90.0, le=90.0)
    target_lon: float = Field(..., ge=-180.0, le=180.0)
    date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class TelemetryPayload(BaseModel):
    mmsi: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    speed: float = Field(default=0.0, ge=0.0)
    heading: float = Field(default=0.0, ge=0.0, lt=360.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_map_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Reads a cached AI map from Redis. Returns None on cache miss."""
    cached = await cache_get(cache_key)
    return cached


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/map", summary="Get combined PFZ + ocean features map for a date")
async def get_mobile_map(
    user: AuthenticatedUser = Depends(get_current_user),
    date: str = Query(
        default=str(dt_date.today()),
        description="Target date YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Returns a combined GeoJSON object containing:
    - Model 1: Potential Fishing Zone predictions (presence probability per cell)
    - Model 2: Ocean fronts and eddies

    Both are read from Redis cache (pre-computed by the nightly ingestion task).
    Response is designed to be a single payload the Flutter map SDK consumes.

    Cache keys:
      model1:pfz:{date}
      model2:ocean_features:{date}
    """
    pfz_data = await _read_map_cache(f"model1:pfz:{date}")
    ocean_data = await _read_map_cache(f"model2:ocean_features:{date}")

    any_cached = pfz_data is not None or ocean_data is not None

    return {
        "date": date,
        "cache_hit": any_cached,
        "pfz": pfz_data or {"type": "FeatureCollection", "features": [], "cache_miss": True},
        "ocean_features": ocean_data or {"fronts": [], "eddies": [], "cache_miss": True},
        "message": None if any_cached else (
            f"No pre-computed data for {date}. The nightly batch job populates this at 02:00 IST."
        ),
    }


@router.post("/route", summary="Get optimized route + time windows (on-demand)")
async def get_route_and_timewindow(
    payload: RouteRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    On-demand route: runs Model 7 (A* route optimization) and Model 8 (solunar
    time windows) in sequence. Implements graceful degradation — if Model 7
    fails, returns a straight-line route with a degradation flag rather than 500.

    This is the primary advisory endpoint for fishermen heading to sea.
    """
    reg = get_model_registry()
    target_date = payload.date or str(dt_date.today())
    result: Dict[str, Any] = {
        "user_id": user.user_id,
        "date": target_date,
        "degraded": False,
        "degraded_reasons": [],
    }

    # ── Model 7: Route Optimization ─────────────────────────────────────────
    if reg.model7 is None:
        result["route"] = {
            "waypoints": [
                {"lat": payload.start_lat, "lon": payload.start_lon},
                {"lat": payload.target_lat, "lon": payload.target_lon},
            ],
            "steps": 2,
        }
        result["degraded"] = True
        result["degraded_reasons"].append("model7_route: not loaded — returning straight-line")
    elif reg.model7.grid_lat is None:
        result["route"] = {
            "waypoints": [
                {"lat": payload.start_lat, "lon": payload.start_lon},
                {"lat": payload.target_lat, "lon": payload.target_lon},
            ],
            "steps": 2,
        }
        result["degraded"] = True
        result["degraded_reasons"].append("model7_route: ocean grid not loaded — straight-line fallback")
    else:
        try:
            route_result = reg.model7.predict(
                payload.start_lat, payload.start_lon,
                payload.target_lat, payload.target_lon,
            )
            result["route"] = {
                "waypoints": route_result.get("route", []),
                "steps": route_result.get("steps", 0),
            }
        except Exception as e:
            logger.warning(f"Model 7 route failed: {e}")
            result["route"] = {"waypoints": [], "steps": 0}
            result["degraded"] = True
            result["degraded_reasons"].append(f"model7_route: {e}")

    # ── Model 8: Time Windows ────────────────────────────────────────────────
    if reg.model8 is None:
        result["time_windows"] = None
        result["degraded"] = True
        result["degraded_reasons"].append("model8_time_window: not loaded")
    else:
        try:
            tw = reg.model8.predict(target_date, payload.target_lat, payload.target_lon)
            result["time_windows"] = tw
        except Exception as e:
            logger.warning(f"Model 8 time window failed: {e}")
            result["time_windows"] = None
            result["degraded"] = True
            result["degraded_reasons"].append(f"model8_time_window: {e}")

    return result


@router.get("/recommendation", summary="Full advisory recommendation (graceful degradation)")
async def get_recommendation(
    user: AuthenticatedUser = Depends(get_current_user),
    vessel_id: str = Query(..., description="MMSI or vessel identifier"),
    current_lat: float = Query(..., ge=-90.0, le=90.0),
    current_lon: float = Query(..., ge=-180.0, le=180.0),
    target_lat: Optional[float] = Query(default=None),
    target_lon: Optional[float] = Query(default=None),
    zone_key: Optional[str] = Query(default=None),
    date: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """
    Orchestrates Models 1, 2, 3, 7, 8, 9 via the Advisory Agent.
    Uses graceful degradation — partial results are returned even if some models fail.
    The `degraded_reasons` array tells the Flutter app which models failed.
    """
    from agents.advisory_agent import build_advisory_agent, FishermanAdvisoryAgent
    from core.redis import get_redis_sync

    date_str = date or str(dt_date.today())
    reg = get_model_registry()

    try:
        redis_client = get_redis_sync()
    except Exception:
        redis_client = None

    agent = build_advisory_agent(reg, redis_client)
    rec = agent.get_recommendation(
        vessel_id=vessel_id,
        date=date_str,
        current_lat=current_lat,
        current_lon=current_lon,
        target_zone_lat=target_lat,
        target_zone_lon=target_lon,
        zone_key=zone_key,
    )
    return FishermanAdvisoryAgent.to_llm_context(rec)


@router.post("/telemetry", summary="Submit live vessel GPS position")
async def submit_telemetry(
    payload: TelemetryPayload,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Accepts a live GPS position from a fisherman's device and:
    1. Updates the Redis Geo-index (GEOADD fleet:live) for real-time fleet tracking
    2. Inserts a row into `vessel_telemetry` in Supabase for historical analytics

    Designed for high-frequency updates (every 30 seconds per vessel).
    Redis update is synchronous; Supabase insert is best-effort.
    """
    from datetime import datetime, timezone
    from core.redis import update_vessel_position_sync
    from core.database import get_supabase

    now = datetime.now(timezone.utc).isoformat()
    meta = {"speed": payload.speed, "heading": payload.heading, "timestamp": now}

    update_vessel_position_sync(payload.mmsi, payload.lat, payload.lon, meta)

    try:
        db = get_supabase()
        db.table("vessel_telemetry").insert({
            "mmsi": payload.mmsi,
            "timestamp": now,
            "lat": payload.lat,
            "lon": payload.lon,
            "speed": payload.speed,
            "heading": payload.heading,
        }).execute()
    except Exception as e:
        logger.warning(f"Supabase telemetry insert failed (Redis updated): {e}")

    return {"status": "ok", "mmsi": payload.mmsi, "timestamp": now}


@router.get("/alerts", summary="Get active safety alerts for this vessel")
async def get_my_alerts(
    user: AuthenticatedUser = Depends(get_current_user),
    mmsi: str = Query(..., description="MMSI of the vessel to query alerts for"),
):
    """
    Returns active safety alerts from `safety_alerts` table filtered to a specific MMSI.
    Used by the Flutter app to show push-notification-style warnings.
    """
    from core.database import get_supabase

    try:
        db = get_supabase()
        result = (
            db.table("safety_alerts")
            .select("*")
            .eq("mmsi", mmsi)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        return {"mmsi": mmsi, "alerts": result.data or [], "count": len(result.data or [])}
    except Exception as e:
        logger.error(f"Failed to fetch alerts for mmsi={mmsi}: {e}")
        raise HTTPException(500, detail="Failed to fetch alerts.")
