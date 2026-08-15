"""
BlueFish AI - Map Routes
=========================
Endpoints for the AI-generated map overlays consumed by the Next.js CesiumJS frontend.

  GET /api/v1/map/pfz?date=YYYY-MM-DD     → PFZ predictions (Model 1)
  GET /api/v1/map/ocean-features?date=... → Fronts & Eddies (Model 2)
  GET /api/v1/map/migration?date=...      → Migration forecast (Model 3)
  GET /api/v1/map/seasonal?date=...       → Seasonal outlook (Model 4)

All map endpoints read from Redis first. The nightly batch job (not in this
file) pre-populates the cache. If the cache is cold, returns a partial
response with a `cache_miss=true` flag.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from core.redis import cache_get, cache_set
from core.model_loader import get_model_registry

logger = logging.getLogger("bluefish.routes.map")

router = APIRouter(prefix="/api/v1/map", tags=["Map Overlays"])


# ── Response schemas ──────────────────────────────────────────────────────────

class MapResponse(BaseModel):
    date: str
    source: str  # "cache" | "live" | "unavailable"
    cache_miss: bool = False
    data: Optional[dict] = None
    message: Optional[str] = None


# ── PFZ (Potential Fishing Zone) ─────────────────────────────────────────────

@router.get("/pfz", response_model=MapResponse, summary="Get PFZ predictions for a date")
async def get_pfz(
    request: Request,
    date: str = Query(
        default=str(date.today()),
        description="Target date in YYYY-MM-DD format",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Returns GeoJSON-compatible PFZ prediction data for the given date.

    Data is pre-computed nightly by the batch job and cached in Redis.
    Cache key: `model1:pfz:{date}`

    If cache is cold, returns a degraded response with cache_miss=True
    rather than triggering a live (expensive) inference run.
    """
    cache_key = f"model1:pfz:{date}"
    cached = await cache_get(cache_key)

    if cached is not None:
        return MapResponse(date=date, source="cache", data=cached)

    # Cache miss — check if model is loaded for on-demand inference fallback
    reg = get_model_registry()
    if reg.model1 is None:
        return MapResponse(
            date=date,
            source="unavailable",
            cache_miss=True,
            message="PFZ model not loaded and cache is cold for this date. Try again after the nightly batch runs.",
        )

    # Return cache miss signal — live inference is too expensive per-request
    # The frontend should show "Data loading..." and retry.
    return MapResponse(
        date=date,
        source="unavailable",
        cache_miss=True,
        message=f"No pre-computed PFZ data for {date}. The nightly batch job populates this cache.",
    )


@router.post("/pfz/compute", summary="[Internal] Compute & cache PFZ for a date")
async def compute_pfz(
    request: Request,
    target_date: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    zone_keys: str = Query(default="gulf_of_mannar,palk_bay,lakshadweep", description="Comma-separated zone keys"),
):
    """
    Triggers on-demand PFZ computation for a specific date and caches the result.
    Called by the nightly batch scheduler — not intended for the public frontend.

    Requires the Model 1 feature inputs to be provided via the request body
    in production. Here it returns a placeholder for the batch job integration.
    """
    reg = get_model_registry()
    if reg.model1 is None:
        raise HTTPException(503, detail="Model 1 (PFZ) not loaded.")

    cache_key = f"model1:pfz:{target_date}"
    # In production, this would iterate over grid points and run model1.predict()
    # for each one, then assemble a GeoJSON FeatureCollection.
    # For now, we return a placeholder to confirm the endpoint is wired.
    placeholder = {
        "type": "FeatureCollection",
        "features": [],
        "note": "Populated by nightly batch job.",
        "date": target_date,
    }
    await cache_set(cache_key, placeholder)
    return {"status": "cached", "date": target_date, "cache_key": cache_key}


# ── Ocean Features (Fronts & Eddies) ─────────────────────────────────────────

@router.get("/ocean-features", response_model=MapResponse, summary="Get fronts & eddies for a date")
async def get_ocean_features(
    request: Request,
    date: str = Query(
        default=str(date.today()),
        description="Target date in YYYY-MM-DD format",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Returns detected ocean fronts and eddies for the given date (Model 2).
    Data is cached nightly after processing the day's NetCDF satellite data.
    Cache key: `model2:ocean_features:{date}`
    """
    cache_key = f"model2:ocean_features:{date}"
    cached = await cache_get(cache_key)

    if cached is not None:
        return MapResponse(date=date, source="cache", data=cached)

    reg = get_model_registry()
    if reg.model2 is None:
        return MapResponse(
            date=date,
            source="unavailable",
            cache_miss=True,
            message="Ocean features model not loaded.",
        )

    return MapResponse(
        date=date,
        source="unavailable",
        cache_miss=True,
        message=f"No pre-computed ocean features for {date}. Populated by nightly batch job.",
    )


# ── Migration Forecast (Model 3 - LSTM) ───────────────────────────────────────

@router.get("/migration", response_model=MapResponse, summary="Get migration forecast for a date")
async def get_migration_forecast(
    request: Request,
    date: str = Query(
        default=str(date.today()),
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Returns the LSTM-based fish migration trajectory forecast.
    Cache key: `model3:migration:{date}`
    """
    cache_key = f"model3:migration:{date}"
    cached = await cache_get(cache_key)

    if cached is not None:
        return MapResponse(date=date, source="cache", data=cached)

    return MapResponse(
        date=date,
        source="unavailable",
        cache_miss=True,
        message=f"No migration forecast cached for {date}.",
    )


# ── Seasonal Outlook (Model 4 - TFT) ─────────────────────────────────────────

@router.get("/seasonal", response_model=MapResponse, summary="Get seasonal fishing outlook")
async def get_seasonal_outlook(
    request: Request,
    date: str = Query(
        default=str(date.today()),
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
):
    """
    Returns the TFT-based seasonal fishing outlook.
    Cache key: `model4:seasonal:{date}`
    """
    cache_key = f"model4:seasonal:{date}"
    cached = await cache_get(cache_key)

    if cached is not None:
        return MapResponse(date=date, source="cache", data=cached)

    reg = get_model_registry()
    if reg.model4 is None:
        return MapResponse(
            date=date,
            source="unavailable",
            cache_miss=True,
            message="TFT seasonal model not loaded (optional dependency).",
        )

    return MapResponse(
        date=date,
        source="unavailable",
        cache_miss=True,
        message=f"No seasonal outlook cached for {date}.",
    )
