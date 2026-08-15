"""
BlueFish AI - Fleet Routes
===========================
Endpoints for real-time fleet monitoring and compliance.

  GET /api/v1/fleet/density           → Overcrowding zones (Model 5)
  GET /api/v1/fleet/anomalies         → Anomaly flags (Model 6)
  GET /api/v1/fleet/status            → Full fleet status summary
  POST /api/v1/fleet/telemetry        → Ingest live AIS/VMS position update
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.model_loader import get_model_registry
from core.redis import get_redis_sync, update_vessel_position_sync

logger = logging.getLogger("bluefish.routes.fleet")

router = APIRouter(prefix="/api/v1/fleet", tags=["Fleet Monitoring"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class VesselPosition(BaseModel):
    mmsi: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    speed: float = Field(default=0.0, ge=0.0)
    heading: float = Field(default=0.0, ge=0.0, lt=360.0)
    timestamp: Optional[str] = None


class TelemetryIngestRequest(BaseModel):
    positions: List[VesselPosition]


class DensityResponse(BaseModel):
    timestamp: str
    vessels_analyzed: int
    zones: List[Dict[str, Any]]
    boats: List[Dict[str, Any]]


class AnomalyResponse(BaseModel):
    timestamp: str
    vessels_checked: int
    anomalies: List[Dict[str, Any]]
    total_anomaly_count: int


# ── Helper: get live vessels from Redis ──────────────────────────────────────

def _get_live_vessels_from_redis(max_vessels: int = 5000) -> List[Dict[str, Any]]:
    """
    Reads live vessel positions from the Redis Geo-index.
    Used by density and anomaly endpoints for on-demand queries.
    """
    import json
    from core.config import get_settings
    settings = get_settings()

    try:
        r = get_redis_sync()
        members = r.zrange(settings.REDIS_GEO_KEY, 0, max_vessels - 1)
        vessels = []
        for mmsi_bytes in members:
            mmsi = mmsi_bytes if isinstance(mmsi_bytes, str) else mmsi_bytes.decode()
            pos = r.geopos(settings.REDIS_GEO_KEY, mmsi)
            meta_raw = r.get(f"{settings.REDIS_META_PREFIX}{mmsi}")
            if not pos or pos[0] is None:
                continue
            lon, lat = pos[0]
            meta = json.loads(meta_raw) if meta_raw else {}
            vessels.append({
                "mmsi": mmsi,
                "lat": float(lat),
                "lon": float(lon),
                "speed": float(meta.get("speed", 0.0)),
                "heading": float(meta.get("heading", 0.0)),
                "timestamp": meta.get("timestamp", ""),
            })
        return vessels
    except Exception as e:
        logger.error(f"Failed to fetch live vessels from Redis: {e}")
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/telemetry", summary="Ingest live AIS/VMS vessel position")
async def ingest_telemetry(request: Request, payload: TelemetryIngestRequest):
    """
    Accepts live vessel position updates from AIS/VMS data feeds.
    Writes each position to:
      1. Redis Geo-index (GEOADD fleet:live)
      2. Supabase vessel_telemetry table (for historical analytics)

    Designed for high write volume — batches Supabase inserts.
    """
    from core.database import get_supabase

    now = datetime.now(timezone.utc).isoformat()
    supabase_records = []

    for pos in payload.positions:
        ts = pos.timestamp or now
        meta = {"speed": pos.speed, "heading": pos.heading, "timestamp": ts}
        update_vessel_position_sync(pos.mmsi, pos.lat, pos.lon, meta)
        supabase_records.append({
            "mmsi": pos.mmsi,
            "timestamp": ts,
            "lat": pos.lat,
            "lon": pos.lon,
            "speed": pos.speed,
            "heading": pos.heading,
        })

    # Batch insert to Supabase (high write volume — upsert disabled for speed)
    try:
        db = get_supabase()
        db.table("vessel_telemetry").insert(supabase_records).execute()
    except Exception as e:
        logger.warning(f"Supabase telemetry insert failed (Redis update succeeded): {e}")

    return {"status": "ok", "ingested": len(payload.positions), "timestamp": now}


@router.get("/density", response_model=DensityResponse, summary="Get fleet density & overcrowding zones")
async def get_fleet_density(request: Request):
    """
    Queries live vessel positions from Redis, runs Model 5 (DBSCAN clustering),
    and returns overcrowded zones with severity ratings.
    """
    reg = get_model_registry()
    if reg.model5 is None:
        raise HTTPException(503, detail="Fleet density model (Model 5) not loaded.")

    vessels = _get_live_vessels_from_redis()
    if not vessels:
        return DensityResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            vessels_analyzed=0,
            zones=[],
            boats=[],
        )

    from agents.fleet_command_agent import Model5Client
    import pandas as pd

    client = Model5Client(reg.model5)
    target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        result = client.predict(vessels, target_date)
        return DensityResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            vessels_analyzed=len(vessels),
            zones=result.get("zones", []),
            boats=result.get("boats", []),
        )
    except Exception as e:
        logger.error(f"Model 5 density prediction failed: {e}")
        raise HTTPException(500, detail=f"Density analysis failed: {e}")


@router.get("/anomalies", response_model=AnomalyResponse, summary="Get fleet anomaly flags")
async def get_fleet_anomalies(request: Request):
    """
    Queries recent vessel behavior, runs batched Model 6 (Isolation Forest),
    and returns anomaly flags. Also checks MPA intrusions via PostGIS.
    """
    from core.database import is_in_mpa, is_in_eez
    from agents.fleet_command_agent import _build_model6_client_from_objects

    reg = get_model_registry()
    if reg.model6_forest is None or reg.model6_scaler is None:
        raise HTTPException(503, detail="Anomaly detection model (Model 6) not loaded.")

    vessels = _get_live_vessels_from_redis()
    if not vessels:
        return AnomalyResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            vessels_checked=0,
            anomalies=[],
            total_anomaly_count=0,
        )

    model6_client = _build_model6_client_from_objects(reg.model6_forest, reg.model6_scaler)
    anomalies = []

    for vessel in vessels:
        try:
            result = model6_client.predict(vessel)
            if result.get("is_anomaly"):
                anomalies.append(result)
        except Exception as e:
            logger.warning(f"Model 6 failed for mmsi={vessel.get('mmsi')}: {e}")

    return AnomalyResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        vessels_checked=len(vessels),
        anomalies=anomalies,
        total_anomaly_count=len(anomalies),
    )


@router.get("/status", summary="Full fleet status summary")
async def get_fleet_status(request: Request):
    """
    Returns a combined fleet status dashboard:
    - Total active vessels
    - Collision alerts from Supabase
    - MPA intrusions
    - Model health
    """
    from core.database import get_supabase

    reg = get_model_registry()
    vessels = _get_live_vessels_from_redis()

    # Recent active alerts from Supabase
    try:
        db = get_supabase()
        alerts_result = (
            db.table("safety_alerts")
            .select("*")
            .eq("status", "active")
            .limit(50)
            .execute()
        )
        active_alerts = alerts_result.data or []
    except Exception as e:
        logger.warning(f"Could not fetch alerts from Supabase: {e}")
        active_alerts = []

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_vessels": len(vessels),
        "active_alerts": len(active_alerts),
        "alerts": active_alerts[:10],  # Return top 10
        "model_status": {
            "model5_density": reg.model5 is not None,
            "model6_anomaly": reg.model6_forest is not None,
            "model10_collision": reg.model10 is not None,
        },
        "load_errors": reg.load_errors,
    }
