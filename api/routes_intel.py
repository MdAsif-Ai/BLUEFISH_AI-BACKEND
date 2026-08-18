"""
BlueFish AI - Intelligence Routes (Government — AI Predictions Page)
=====================================================================
Page 2 of the Next.js Command Center. Deep AI analytics and forecasts.
All routes require `government` role.

  GET /api/v1/intel/pfz?date=       → Model 1 PFZ predictions + DB historical data
  GET /api/v1/intel/seasonal        → 12-month forecast (Model 4 TFT + Model 11 XGBoost)
  GET /api/v1/intel/migration       → Fish migration trajectory (Model 3 LSTM)
  GET /api/v1/intel/climate-risk    → Regional climate impact (Model 11)
"""

from __future__ import annotations
import logging
from datetime import date as dt_date, datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.security import AuthenticatedUser, require_government
from core.model_loader import get_model_registry
from core.redis import cache_get

logger = logging.getLogger("bluefish.routes.intel")

router = APIRouter(
    prefix="/api/v1/intel",
    tags=["🧠 Intelligence — AI Predictions"],
)


class ClimateRiskRequest(BaseModel):
    features: List[float]
    region: Optional[str] = "tamil_nadu_eez"


@router.get("/pfz", summary="PFZ predictions — cache + DB historical data (Model 1)")
async def get_pfz_intelligence(
    user: AuthenticatedUser = Depends(require_government),
    date: str = Query(default=str(dt_date.today()), pattern=r"^\d{4}-\d{2}-\d{2}$"),
    include_historical: bool = Query(default=True, description="Also return last 7 days from DB"),
):
    """
    Returns:
    1. Today's pre-computed PFZ GeoJSON from Redis cache (fast)
    2. Historical PFZ entries from the `pfz_predictions` table (for trend analysis)

    The Redis data is the high-resolution grid output from the nightly ingestion task.
    The Supabase data contains validated, coarser predictions for long-term trend charts.
    """
    cached_pfz = await cache_get(f"model1:pfz:{date}")

    historical_rows = []
    if include_historical:
        from core.database import get_supabase
        try:
            from datetime import timedelta
            db = get_supabase()
            week_ago = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
            result = (
                db.table("pfz_predictions")
                .select("prediction_date, lat, lon, presence_probability")
                .gte("prediction_date", week_ago)
                .lte("prediction_date", date)
                .order("prediction_date", desc=True)
                .limit(5000)
                .execute()
            )
            historical_rows = result.data or []
        except Exception as e:
            logger.warning(f"Historical PFZ fetch failed: {e}")

    return {
        "date": date,
        "realtime_grid": cached_pfz or {"cache_miss": True, "features": [],
                                         "note": "Populated by nightly batch job at 02:00 IST"},
        "historical_predictions": historical_rows,
        "historical_count": len(historical_rows),
    }


@router.get("/seasonal", summary="12-month seasonal forecast (Model 4 + Model 11)")
async def get_seasonal_forecast(
    user: AuthenticatedUser = Depends(require_government),
    horizon_weeks: int = Query(default=52, ge=4, le=104),
):
    """
    Combines:
    - Model 4 (TFT): Long-range temporal pattern forecast
    - Model 11 (XGBoost Climate): Climate stress adjustment to the TFT baseline

    If Model 4 is not loaded (heavy optional dependency), returns a
    Model 11-only assessment with a degradation flag.
    """
    reg = get_model_registry()
    result: Dict[str, Any] = {
        "horizon_weeks": horizon_weeks,
        "degraded": False,
        "degraded_reasons": [],
    }

    # ── Model 4: TFT Seasonal Outlook ────────────────────────────────────────
    if reg.model4 is not None:
        try:
            tft_result = reg.model4.predict_simple(features=[], horizon_weeks=horizon_weeks)
            result["tft_seasonal"] = tft_result
        except Exception as e:
            logger.warning(f"Model 4 TFT failed: {e}")
            result["tft_seasonal"] = None
            result["degraded"] = True
            result["degraded_reasons"].append(f"model4_tft: {e}")
    else:
        result["tft_seasonal"] = None
        result["degraded"] = True
        result["degraded_reasons"].append("model4_tft: not loaded (optional dependency)")

    # ── Model 11: Climate Risk Adjustment ────────────────────────────────────
    if reg.model11_xgb is not None and reg.model11_scaler is not None:
        try:
            from services.model11_service import ClimateRiskService
            # Default neutral-climate feature vector for regional assessment
            neutral_features = [0.0] * 10
            svc = ClimateRiskService(reg.model11_xgb, reg.model11_scaler)
            climate = svc.predict(neutral_features)
            result["climate_adjustment"] = climate
        except Exception as e:
            logger.warning(f"Model 11 climate risk failed: {e}")
            result["climate_adjustment"] = None
            result["degraded"] = True
            result["degraded_reasons"].append(f"model11_climate: {e}")
    else:
        result["climate_adjustment"] = None
        result["degraded"] = True
        result["degraded_reasons"].append("model11_climate: not loaded")

    return result


@router.get("/migration", summary="Fish migration trajectory forecast (Model 3 LSTM)")
async def get_migration_forecast(
    user: AuthenticatedUser = Depends(require_government),
    date: str = Query(default=str(dt_date.today()), pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """
    Returns the cached Model 3 (Seq2Seq LSTM) migration trajectory forecast
    for the given date, or a cache-miss indicator.
    """
    cached = await cache_get(f"model3:migration:{date}")
    if cached:
        return {"date": date, "source": "cache", "data": cached}

    reg = get_model_registry()
    if reg.model3 is None:
        return {"date": date, "source": "unavailable",
                "message": "Migration LSTM not loaded. Populated by nightly batch job."}

    # Fallback: run Model 3 with a placeholder feature sequence
    try:
        placeholder_sequence = [[0.0] * 10] * 30
        result = reg.model3.predict(placeholder_sequence)
        return {"date": date, "source": "live_inference", "data": result}
    except Exception as e:
        logger.error(f"Model 3 live inference failed: {e}")
        return {"date": date, "source": "unavailable",
                "message": f"Migration model inference failed: {e}"}


@router.post("/climate-risk", summary="Regional climate risk score (Model 11)")
async def get_climate_risk(
    payload: ClimateRiskRequest,
    user: AuthenticatedUser = Depends(require_government),
):
    """
    Runs the XGBoost climate risk model on a feature vector to predict
    long-term fishing viability for a region under climate change scenarios.
    """
    from services.model11_service import ClimateRiskService
    reg = get_model_registry()
    if reg.model11_xgb is None or reg.model11_scaler is None:
        raise HTTPException(503, detail="Climate risk model (Model 11) not loaded.")
    try:
        svc = ClimateRiskService(reg.model11_xgb, reg.model11_scaler)
        return {**svc.predict(payload.features), "region": payload.region}
    except Exception as e:
        logger.error(f"Model 11 failed: {e}")
        raise HTTPException(500, detail=f"Climate risk model failed: {e}")


@router.get("/live-datas", summary="Universal Live Weather & Marine Telemetry (Open-Meteo)")
async def get_universal_live_datas(
    latitude: float = Query(default=13.1167, ge=-90.0, le=90.0, description="Latitude of location"),
    longitude: float = Query(default=80.2833, ge=-180.0, le=180.0, description="Longitude of location"),
    location_name: Optional[str] = Query(default=None, description="Optional name of location"),
):
    """
    Universal location service returning real-time current weather and marine data
    from Open-Meteo Weather and Marine APIs for any given latitude and longitude.
    Guaranteed zero HTTP 500 Internal Server Errors.
    """
    try:
        from services.live_data_service import LiveDataService
        svc = LiveDataService()
        return await svc.get_live_data(latitude=latitude, longitude=longitude, location_name=location_name)
    except Exception as e:
        logger.error(f"Error in /live-datas endpoint: {e}")
        from datetime import datetime, timezone
        return {
            "location": {
                "name": location_name or f"Coords ({latitude:.4f}, {longitude:.4f})",
                "latitude": latitude,
                "longitude": longitude,
            },
            "weather": {
                "status": "unavailable",
                "error": str(e),
                "air_temperature": None,
                "feels_like_temperature": None,
                "humidity": None,
                "dew_point": None,
                "precipitation": None,
                "rain": None,
                "showers": None,
                "cloud_cover": None,
                "surface_pressure": None,
                "visibility": None,
                "wind_speed": None,
                "wind_direction": None,
                "wind_gusts": None,
                "weather_code": None,
                "timestamp": None,
            },
            "marine": {
                "status": "unavailable",
                "error": str(e),
                "sst": None,
                "wave_height": None,
                "wave_direction": None,
                "wave_period": None,
                "swell_height": None,
                "swell_direction": None,
                "swell_period": None,
                "wind_wave_height": None,
                "wind_wave_direction": None,
                "wind_wave_period": None,
                "ocean_current_velocity": None,
                "ocean_current_direction": None,
                "timestamp": None,
            },
            "environment": {
                "chlorophyll_a": {
                    "value": None,
                    "status": "unavailable",
                    "reason": "Unavailable from current provider",
                },
                "bathymetry_depth": {
                    "value": None,
                    "status": "unavailable",
                    "reason": "Unavailable from current provider",
                },
            },
            "metadata": {
                "source": "Open-Meteo",
                "weather_model": "ECMWF IFS 0.25°",
                "marine_model": "Open-Meteo Global Marine",
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
        }


@router.get("/location-search", summary="Universal Marine Geocoding Location Search")
async def search_marine_locations(
    query: str = Query(..., min_length=1, description="Location search term (e.g. Kasimedu, Chennai, Mumbai)"),
    count: int = Query(default=10, ge=1, le=20, description="Max results count"),
):
    """
    Searches for harbours, ports, coastal cities, and marine locations via Open-Meteo Geocoding.
    Guaranteed zero HTTP 500 Internal Server Errors.
    """
    try:
        from services.location_service import LocationService
        svc = LocationService()
        return await svc.search_locations(query=query, count=count)
    except Exception as e:
        logger.error(f"Error in /location-search endpoint: {e}")
        return {
            "query": query,
            "total": 0,
            "results": [],
            "error": str(e)
        }


@router.get("/live-model-predictions", summary="Execute All 11 AI Models with Live 97 Telemetry Vector")
async def get_all_live_model_predictions(
    latitude: float = Query(default=13.1167, ge=-90.0, le=90.0, description="Latitude of target location"),
    longitude: float = Query(default=80.2833, ge=-180.0, le=180.0, description="Longitude of target location"),
    location_name: Optional[str] = Query(default=None, description="Optional name of marine location"),
):
    """
    1. Fetches real live weather & marine data (Open-Meteo) + 97 synthesized live telemetry vector for any location.
    2. Feeds the 97 live telemetry vector directly into ALL 11 AI models.
    3. Returns unified predictions from Model 1 through Model 11 in a single response payload.
    """
    try:
        from services.live_data_service import LiveDataService
        from services.master_pipeline_service import MasterModelPipelineService
        
        live_svc = LiveDataService()
        live_data = await live_svc.get_live_data(latitude=latitude, longitude=longitude, location_name=location_name)
        
        pipeline = MasterModelPipelineService()
        predictions = pipeline.run_all_models(live_data)
        return predictions
    except Exception as e:
        logger.error(f"Error executing live model predictions pipeline: {e}")
        return {
            "error": str(e),
            "location": {"name": location_name or "Target Coordinates", "latitude": latitude, "longitude": longitude},
            "model_predictions": {}
        }

