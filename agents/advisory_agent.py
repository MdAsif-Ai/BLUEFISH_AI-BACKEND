"""
BlueFish AI - Fisherman Advisory Agent (FastAPI-integrated)
============================================================
Thin FastAPI adapter that wires the existing FishermanAdvisoryAgent
(from AGENTS/agent_1.py) to the model registry and Redis cache.

This module should be imported by main.py to create the singleton agent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ── Import existing agent code from AGENTS/ directory ────────────────────────
_agents_dir = Path(__file__).parent.parent.parent / "AGENTS"
sys.path.insert(0, str(_agents_dir))

from agent_1 import (  # noqa: E402
    FishermanAdvisoryAgent,
    InMemoryPredictionCache,
    Model1Client,
    Model9Client,
    PredictionCache,
)

logger = logging.getLogger("bluefish.advisory_agent_wire")

# ── Re-export for convenience ─────────────────────────────────────────────────
__all__ = [
    "FishermanAdvisoryAgent",
    "InMemoryPredictionCache",
    "Model1Client",
    "Model9Client",
    "PredictionCache",
    "build_advisory_agent",
]


class RedisPredictionCache(PredictionCache):
    """
    Production cache backend backed by Redis.
    Wraps the sync Redis client for use in the advisory agent.
    """

    def __init__(self, redis_client, ttl_seconds: int = 86400):
        self._r = redis_client
        self._ttl = ttl_seconds

    def get(self, key: str):
        import json
        try:
            raw = self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"RedisPredictionCache.get failed key={key}: {e}")
            return None

    def set(self, key: str, value, ttl_seconds: int = 86400):
        import json
        try:
            self._r.setex(key, ttl_seconds, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"RedisPredictionCache.set failed key={key}: {e}")


def build_advisory_agent(model_registry, redis_client=None) -> FishermanAdvisoryAgent:
    """
    Constructs the FishermanAdvisoryAgent wired to the loaded model registry
    and Redis cache. Called once at startup.
    """
    from core.config import get_settings
    settings = get_settings()

    if redis_client is not None:
        cache = RedisPredictionCache(redis_client, ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS)
    else:
        logger.warning("Redis not available — using in-memory cache for advisory agent.")
        cache = InMemoryPredictionCache()

    # Model 9 (KMeans) wrapper
    model9_client: Optional[Model9Client] = None
    if model_registry.model9_kmeans is not None:
        try:
            import pickle, tempfile, os
            # Persist to a temp path so Model9Client can load it
            tmp_path = Path(__file__).parent.parent / "tmp" / "kmeans_model.pkl"
            model9_client = Model9Client(str(tmp_path))
        except Exception as e:
            logger.warning(f"Could not initialize Model9Client: {e}")

    return FishermanAdvisoryAgent(
        cache=cache,
        model1_client=model_registry.model1,
        model2_service=model_registry.model2,
        model7_service=model_registry.model7,
        model8_service=model_registry.model8,
        model9_client=model9_client,
    )
