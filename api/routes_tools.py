"""
BlueFish AI - Tools Routes
===========================
On-demand fisherman advisory tools.

  POST /api/v1/tools/optimize-route      → Fuel-efficient route (Model 7)
  GET  /api/v1/tools/time-window         → Solunar feeding times (Model 8)
  GET  /api/v1/tools/recommendation      → Full advisory recommendation
  POST /api/v1/tools/climate-risk        → Climate impact score (Model 11)
"""

from __future__ import annotations
import logging
from datetime import date as dt_date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.model_loader import get_model_registry

logger = logging.getLogger("bluefish.routes.tools")
router = APIRouter(prefix="/api/v1/tools", tags=["Fisherman Tools"])


class RouteRequest(BaseModel):
    start_lat: float = Field(..., ge=-90.0, le=90.0)
    start_lon: float = Field(..., ge=-180.0, le=180.0)
    target_lat: float = Field(..., ge=-90.0, le=90.0)
    target_lon: float = Field(..., ge=-180.0, le=180.0)


class ClimateRiskRequest(BaseModel):
    features: List[float]


@router.post("/optimize-route", summary="Optimal fuel-efficient route (Model 7)")
async def optimize_route(request: Request, payload: RouteRequest):
    """
    Runs A* route optimization weighted by ocean currents and bathymetry.
    Falls back to straight-line if grid data not loaded.
    """
    reg = get_model_registry()
    if reg.model7 is None:
        raise HTTPException(503, detail="Route optimization model (Model 7) not loaded.")

    model7 = reg.model7

    if model7.grid_lat is None:
        logger.warning("Model 7 grid not loaded — returning straight-line fallback.")
        return {
            "start": {"lat": payload.start_lat, "lon": payload.start_lon},
            "destination": {"lat": payload.target_lat, "lon": payload.target_lon},
            "route": [
                {"lat": payload.start_lat, "lon": payload.start_lon},
                {"lat": payload.target_lat, "lon": payload.target_lon},
            ],
            "steps": 2,
            "note": "Straight-line route — ocean current grid not loaded.",
        }

    try:
        result = model7.predict(payload.start_lat, payload.start_lon, payload.target_lat, payload.target_lon)
        return {
            "start": {"lat": payload.start_lat, "lon": payload.start_lon},
            "destination": {"lat": payload.target_lat, "lon": payload.target_lon},
            **result,
        }
    except Exception as e:
        logger.error(f"Route optimization failed: {e}")
        raise HTTPException(500, detail=f"Route optimization failed: {e}")


@router.get("/time-window", summary="Solunar feeding time windows (Model 8)")
async def get_time_window(
    request: Request,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    date: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    """
    Pure solunar math — always available regardless of satellite data.
    Returns major/minor feeding windows, moon phase, and daily rating.
    """
    reg = get_model_registry()
    if reg.model8 is None:
        raise HTTPException(503, detail="Time window model (Model 8) not loaded.")
    try:
        return reg.model8.predict(date, lat, lon)
    except Exception as e:
        logger.error(f"Time window prediction failed: {e}")
        raise HTTPException(500, detail=f"Time window calculation failed: {e}")


@router.get("/recommendation", summary="Full advisory recommendation (Models 1-3, 7-9)")
async def get_recommendation(
    request: Request,
    vessel_id: str = Query(...),
    current_lat: float = Query(..., ge=-90.0, le=90.0),
    current_lon: float = Query(..., ge=-180.0, le=180.0),
    target_lat: Optional[float] = Query(default=None),
    target_lon: Optional[float] = Query(default=None),
    zone_key: Optional[str] = Query(default=None),
    date: Optional[str] = Query(default=None, regex=r"^\d{4}-\d{2}-\d{2}$"),
):
    """
    Orchestrates Models 1, 2, 3, 7, 8, 9 via the Advisory Agent.
    Implements graceful degradation with a `degraded_reasons` array.
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


@router.post("/climate-risk", summary="Climate risk score (Model 11 - XGBoost)")
async def get_climate_risk(request: Request, payload: ClimateRiskRequest):
    """Runs XGBoost climate impact model on a feature vector."""
    from services.model11_service import ClimateRiskService
    reg = get_model_registry()
    if reg.model11_xgb is None or reg.model11_scaler is None:
        raise HTTPException(503, detail="Climate risk model (Model 11) not loaded.")
    try:
        service = ClimateRiskService(reg.model11_xgb, reg.model11_scaler)
        return service.predict(payload.features)
    except Exception as e:
        logger.error(f"Climate risk prediction failed: {e}")
        raise HTTPException(500, detail=f"Climate risk model failed: {e}")
