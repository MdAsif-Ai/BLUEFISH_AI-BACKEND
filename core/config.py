"""
BlueFish AI - Core Configuration
=================================
All settings are loaded from environment variables.
On Render.com, set these in the service's Environment tab.
For local development, use a .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "BlueFish AI Backend"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = "INFO"

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: str = Field(..., description="Your Supabase project URL")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(..., description="Service role key - bypasses RLS")
    # Found in: Supabase Dashboard → Project Settings → API → JWT Settings → JWT Secret
    SUPABASE_JWT_SECRET: str = Field(..., description="Used to verify user JWTs from the frontend")
    MOCK_JWT_SECRET: str = Field(default="dev-mock-secret-for-local-testing", description="Used for local PyJWT mock auth")

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    REDIS_CACHE_TTL_SECONDS: int = Field(default=86400, description="24h cache TTL for daily AI maps")
    REDIS_GEO_KEY: str = "fleet:live"
    REDIS_META_PREFIX: str = "fleet:meta:"

    # ── Supabase Storage ──────────────────────────────────────────────────────
    ML_MODELS_BUCKET: str = "ml-models"

    # ── Model 1 Feature Contract ──────────────────────────────────────────────
    # CRITICAL: Order MUST match exactly what the ONNX model was trained on.
    MODEL1_FEATURE_ORDER: List[str] = [
        "month", "dayofyear", "ONI_Value", "sst", "salinity",
        "current_east", "current_north", "chlorophyll",
        "current_speed", "current_direction_deg",
    ]

    # ── Fleet Monitoring ──────────────────────────────────────────────────────
    FLEET_POLL_INTERVAL_SECONDS: int = 60
    FLEET_MAX_VESSELS_PER_CYCLE: int = 20_000

    # ── Collision Detection Thresholds ────────────────────────────────────────
    COLLISION_CPA_THRESHOLD_KM: float = 0.5
    COLLISION_TCPA_THRESHOLD_MIN: float = 15.0

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    ANOMALY_SCORE_THRESHOLD: float = -0.1

    # ── Retraining ────────────────────────────────────────────────────────────
    RETRAINING_MIN_NEW_FEEDBACK_ROWS: int = 200

    # ── Digital Twin ─────────────────────────────────────────────────────────
    DIGITAL_TWIN_MAX_DAYS: int = 90

    # ── Celery ───────────────────────────────────────────────────────────────
    # Uses Redis as both broker and result backend.
    # Same REDIS_URL is fine; Celery uses different key prefixes automatically.
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", description="Celery broker (Redis DB 1)")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", description="Celery results (Redis DB 2)")

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["*"]

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
