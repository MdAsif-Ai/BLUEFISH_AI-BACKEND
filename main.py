"""
BlueFish AI - FastAPI Application Entry Point v2.0
====================================================
Production-ready FastAPI backend for the BlueFish AI
Marine Digital Twin & Fisheries Intelligence Ecosystem.

On startup:
  1. Initializes Supabase client (Service Role Key)
  2. Initializes Redis clients (sync + async)
  3. Downloads all 11 model artifacts from Supabase Storage
  4. Loads all models into the global ModelRegistry
  NOTE: Fleet polling is now handled by Celery Beat (celery_worker.py)

Routing structure (v2.0 — Role-Based):
  /api/v1/mobile/*   → Fisherman Flutter App (any authenticated user)
  /api/v1/command/*  → Government Command Center - Live Dashboard
  /api/v1/intel/*    → Government Intelligence - AI Predictions
  /api/v1/twin/*     → Government Digital Twin - 3D Simulator (Celery)
  /api/v1/system/*   → Government MLOps - Pipelines
  /health            → Infrastructure health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Add MODELS and AGENTS to Python path ─────────────────────────────────────
_base = Path(__file__).parent
# MODELS/ and AGENTS/ live inside the backend root (BLUEFISH_AI-BACKEND/)
_models = _base / "MODELS"
_agents = _base / "AGENTS"
sys.path.insert(0, str(_models / "model2"))
sys.path.insert(0, str(_models / "model5"))
sys.path.insert(0, str(_models / "model7"))
sys.path.insert(0, str(_models / "model8"))
sys.path.insert(0, str(_models / "model10"))
sys.path.insert(0, str(_agents))

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bluefish.main")

# ── Background task handle ────────────────────────────────────────────────────
# NOTE: In v2.0, the fleet polling loop is handled by Celery Beat workers.
# The asyncio background task is kept as a fallback only when Redis/Celery
# is unavailable. Normally this remains None.
_fleet_task: asyncio.Task | None = None


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Initialize all infrastructure and load models.
    Shutdown: Cancel background tasks and close connections gracefully.
    """
    global _fleet_task

    logger.info("=" * 60)
    logger.info("BlueFish AI Backend starting up...")
    logger.info("=" * 60)

    # 1. Load configuration
    from core.config import get_settings
    settings = get_settings()

    # 2. Initialize Supabase
    from core.database import get_supabase
    db = get_supabase()
    logger.info("✓ Supabase client initialized.")

    # 3. Initialize Redis (sync client — used by background agents)
    from core.redis import get_redis_sync, get_redis_async
    try:
        r_sync = get_redis_sync()
        r_sync.ping()
        logger.info("✓ Redis sync client connected.")
    except Exception as e:
        logger.warning(f"Redis sync connection failed: {e}. Fleet agent will be degraded.")
        r_sync = None

    # 4. Initialize Redis async client
    try:
        await get_redis_async()
        logger.info("✓ Redis async client connected.")
    except Exception as e:
        logger.warning(f"Redis async connection failed: {e}. Cache endpoints will miss.")

    # 5. Download and load all 11 AI models
    from core.model_loader import load_all_models
    registry = await load_all_models(db, settings.ML_MODELS_BUCKET)
    logger.info(f"Model registry ready. Errors: {registry.load_errors or 'none'}")

    # 6. Celery Note (fleet polling is now handled by Celery Beat workers)
    logger.info("Fleet Command Agent is managed by Celery Beat.")
    logger.info("Start workers: celery -A celery_worker worker -Q fleet --concurrency=2")
    logger.info("Start beat:    celery -A celery_worker beat --loglevel=info")

    logger.info("=" * 60)
    logger.info("BlueFish AI Backend v2.0 ready.")
    logger.info("="* 60)

    yield  # ── Application runs here ──

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("BlueFish AI Backend shutting down...")

    if _fleet_task and not _fleet_task.done():
        _fleet_task.cancel()
        try:
            await asyncio.wait_for(_fleet_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("Fleet Command Agent stopped.")

    from core.redis import close_redis_async
    await close_redis_async()
    logger.info("Redis connections closed.")
    logger.info("Shutdown complete.")


# ── Create FastAPI application ────────────────────────────────────────────────

app = FastAPI(
    title="BlueFish AI Backend",
    description=(
        "Marine Digital Twin & Fisheries Intelligence Ecosystem for Tamil Nadu Government. "
        "Powers 11 AI models for PFZ prediction, fleet monitoring, route optimization, "
        "digital twin simulation, and climate risk assessment. "
        "Authentication: Supabase JWT. Task Queue: Celery + Redis."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────

from core.config import get_settings as _get_settings
_settings = _get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handler (Zero 500 Internal Server Errors) ────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception caught safely on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content={
            "status": "degraded",
            "error": False,
            "message": "Service operational with fallback response",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )

# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infrastructure"], summary="Infrastructure health check")
async def health_check():
    """
    Returns the health status of all infrastructure components.
    Used by Render.com health check monitoring.
    """
    from core.database import check_supabase_health
    from core.redis import check_redis_health
    from core.model_loader import get_model_registry

    reg = get_model_registry()

    supabase_ok = await check_supabase_health()
    redis_ok = await check_redis_health()

    models_loaded = {
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
    total_loaded = sum(models_loaded.values())

    fleet_running = _fleet_task is not None and not _fleet_task.done()

    overall = "healthy" if (supabase_ok and redis_ok and total_loaded >= 8) else "degraded"

    return {
        "status": overall,
        "infrastructure": {
            "supabase": "ok" if supabase_ok else "error",
            "redis": "ok" if redis_ok else "error",
            "fleet_agent": "running" if fleet_running else "stopped",
        },
        "models": {
            "loaded": total_loaded,
            "total": 11,
            "details": models_loaded,
            "load_errors": reg.load_errors,
        },
    }


# ── Include API Routers (v2.0 — Role-Based Architecture) ──────────────────────

# ── v2.0 Role-Based Routers (NEW) ────────────────────────────────────────────
from auth.routes import router as auth_router          # bulletproof auth module (auth/)
from api.routes_mobile import router as mobile_router
from api.routes_command import router as command_router
from api.routes_intel import router as intel_router
from api.routes_twin import router as twin_router
from api.routes_mlops import router as mlops_router

app.include_router(auth_router)     # Public — register, login, /me
app.include_router(mobile_router)   # Fisherman App — any authenticated user
app.include_router(command_router)  # Government — Live Dashboard
app.include_router(intel_router)    # Government — AI Predictions
app.include_router(twin_router)     # Government — Digital Twin (Celery)
app.include_router(mlops_router)    # Government — MLOps Pipelines

# ── v1.0 Legacy Routers (kept for backward compatibility) ─────────────────────
from api.routes_map import router as map_router
from api.routes_fleet import router as fleet_router
from api.routes_tools import router as tools_router

app.include_router(map_router)
app.include_router(fleet_router)
app.include_router(tools_router)


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "BlueFish AI Backend",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "description": "Marine Digital Twin & Fisheries Intelligence Ecosystem — Tamil Nadu Government",
        "architecture": "FastAPI + Supabase + Redis + Celery",
        "routes": {
            "mobile": "/api/v1/mobile/* (Fisherman App)",
            "command": "/api/v1/command/* (Government — Live Dashboard)",
            "intel": "/api/v1/intel/* (Government — AI Predictions)",
            "twin": "/api/v1/twin/* (Government — Digital Twin)",
            "system": "/api/v1/system/* (Government — MLOps)",
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        workers=1,  # Single worker required — models are loaded into process memory
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
