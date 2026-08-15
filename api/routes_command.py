"""
BlueFish AI - Command Center Routes (Government — Live Dashboard)
=================================================================
All routes require `government` role.

  GET /api/v1/command/fleet/density    → Overcrowding zones (Model 5)
  GET /api/v1/command/fleet/anomalies  → Anomaly & compliance flags (Model 6)
  GET /api/v1/command/fleet/status     → Combined fleet health dashboard
  GET /api/v1/command/fleet/collisions → Active collision alerts
  POST /api/v1/command/alert/resolve   → Mark alert resolved
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.security import AuthenticatedUser, require_government
from core.model_loader import get_model_registry

logger = logging.getLogger("bluefish.routes.command")

router = APIRouter(
    prefix="/api/v1/command",
    tags=["🖥️ Command Center"],
    dependencies=[Depends(require_government)],
)


class ResolveAlertRequest(BaseModel):
    alert_id: str
    resolved_by: str


def _get_live_vessels(max_vessels: int = 5000) -> List[Dict[str, Any]]:
    import json
    from core.config import get_settings
    from core.redis import get_redis_sync

    settings = get_settings()
    try:
        r = get_redis_sync()
        members = r.zrange(settings.REDIS_GEO_KEY, 0, max_vessels - 1)
        vessels = []
        for mmsi in members:
            mmsi = mmsi if isinstance(mmsi, str) else mmsi.decode()
            pos = r.geopos(settings.REDIS_GEO_KEY, mmsi)
            meta_raw = r.get(f"{settings.REDIS_META_PREFIX}{mmsi}")
            if not pos or pos[0] is None:
                continue
            lon, lat = pos[0]
            meta = json.loads(meta_raw) if meta_raw else {}
            vessels.append({"mmsi": mmsi, "lat": float(lat), "lon": float(lon),
                            "speed": float(meta.get("speed", 0.0)),
                            "heading": float(meta.get("heading", 0.0)),
                            "timestamp": meta.get("timestamp", "")})
        return vessels
    except Exception as e:
        logger.error(f"Redis vessel fetch failed: {e}")
        return []


@router.get("/fleet/density", summary="Fleet overcrowding zones (Model 5)")
async def get_fleet_density(user: AuthenticatedUser = Depends(require_government)):
    """Reads live GPS from Redis Geo-index and runs DBSCAN (Model 5)."""
    reg = get_model_registry()
    if reg.model5 is None:
        raise HTTPException(503, detail="Fleet density model not loaded.")

    vessels = _get_live_vessels()
    if not vessels:
        return {"timestamp": datetime.now(timezone.utc).isoformat(),
                "vessels_analyzed": 0, "zones": [], "boats": []}

    from agents.fleet_command_agent import Model5Client
    client = Model5Client(reg.model5)
    result = client.predict(vessels, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            "vessels_analyzed": len(vessels), **result}


@router.get("/fleet/anomalies", summary="Anomaly & compliance flags (Model 6)")
async def get_fleet_anomalies(user: AuthenticatedUser = Depends(require_government)):
    """Fetches persisted anomalies from Supabase + runs live Model 6 scan."""
    from core.database import get_supabase
    db = get_supabase()

    try:
        result = (db.table("safety_alerts").select("*")
                  .in_("alert_type", ["anomaly", "mpa_intrusion"])
                  .eq("status", "active").order("created_at", desc=True).limit(100).execute())
        persisted = result.data or []
    except Exception as e:
        logger.error(f"Supabase anomaly fetch failed: {e}")
        persisted = []

    live_anomalies = []
    reg = get_model_registry()
    if reg.model6_forest is not None and reg.model6_scaler is not None:
        from agents.fleet_command_agent import _build_model6_client_from_objects
        model6 = _build_model6_client_from_objects(reg.model6_forest, reg.model6_scaler)
        for v in _get_live_vessels(max_vessels=500):
            try:
                r = model6.predict(v)
                if r.get("is_anomaly"):
                    live_anomalies.append(r)
            except Exception:
                pass

    return {"timestamp": datetime.now(timezone.utc).isoformat(),
            "persisted_anomalies": persisted, "live_anomalies": live_anomalies,
            "total": len(persisted) + len(live_anomalies)}


@router.get("/fleet/collisions", summary="Active collision alerts")
async def get_collision_alerts(user: AuthenticatedUser = Depends(require_government)):
    from core.database import get_supabase
    db = get_supabase()
    try:
        result = (db.table("safety_alerts").select("*")
                  .eq("alert_type", "collision").eq("status", "active")
                  .order("created_at", desc=True).limit(50).execute())
        return {"timestamp": datetime.now(timezone.utc).isoformat(),
                "alerts": result.data or [], "count": len(result.data or [])}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch collision alerts: {e}")


@router.get("/fleet/status", summary="Combined fleet health dashboard")
async def get_fleet_status(user: AuthenticatedUser = Depends(require_government)):
    from core.database import get_supabase
    reg = get_model_registry()
    vessels = _get_live_vessels()
    try:
        db = get_supabase()
        ar = db.table("safety_alerts").select("alert_type", count="exact").eq("status", "active").execute()
        active_alerts = ar.data or []
        alert_count = ar.count or 0
    except Exception:
        active_alerts, alert_count = [], 0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fleet": {"active_vessels": len(vessels), "source": "redis_geo_index"},
        "alerts": {"active_count": alert_count,
                   "breakdown": {
                       "collision": sum(1 for a in active_alerts if a.get("alert_type") == "collision"),
                       "anomaly": sum(1 for a in active_alerts if a.get("alert_type") == "anomaly"),
                       "mpa_intrusion": sum(1 for a in active_alerts if a.get("alert_type") == "mpa_intrusion")}},
        "models": {"model5": reg.model5 is not None, "model6": reg.model6_forest is not None,
                   "model10": reg.model10 is not None},
    }


@router.post("/alert/resolve", summary="Mark safety alert as resolved")
async def resolve_alert(payload: ResolveAlertRequest, user: AuthenticatedUser = Depends(require_government)):
    from core.database import get_supabase
    db = get_supabase()
    try:
        result = (db.table("safety_alerts")
                  .update({"status": "resolved", "details": {"resolved_by": payload.resolved_by}})
                  .eq("id", payload.alert_id).execute())
        if not result.data:
            raise HTTPException(404, detail=f"Alert {payload.alert_id} not found.")
        return {"status": "resolved", "alert_id": payload.alert_id, "resolved_by": payload.resolved_by}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to resolve alert: {e}")
