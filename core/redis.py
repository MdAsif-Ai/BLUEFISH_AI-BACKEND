"""
BlueFish AI - Redis Client & Cache Layer
==========================================
Provides:
  - Singleton async Redis client (aioredis-compatible via redis.asyncio)
  - Sync Redis client for background agents
  - JSON-aware cache helpers (get/set with automatic serialization)
  - Pub/Sub publisher for real-time safety alerts
  - Geo-index helpers for live vessel tracking

Usage (async):
    from core.redis import get_redis_async, cache_get, cache_set

Usage (sync, for background agents):
    from core.redis import get_redis_sync
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Optional

import redis
import redis.asyncio as aioredis

from core.config import get_settings

logger = logging.getLogger("bluefish.redis")

# ── Sync client (for background agents / APScheduler jobs) ────────────────────

_sync_client: Optional[redis.Redis] = None


def get_redis_sync() -> redis.Redis:
    global _sync_client
    if _sync_client is None:
        settings = get_settings()
        logger.info(f"Connecting to Redis (sync): {settings.REDIS_URL}")
        _sync_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        logger.info("Redis sync client ready.")
    return _sync_client


# ── Async client (for FastAPI request handlers) ───────────────────────────────

_async_client: Optional[aioredis.Redis] = None


async def get_redis_async() -> aioredis.Redis:
    global _async_client
    if _async_client is None:
        settings = get_settings()
        logger.info(f"Connecting to Redis (async): {settings.REDIS_URL}")
        _async_client = await aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        logger.info("Redis async client ready.")
    return _async_client


async def close_redis_async() -> None:
    global _async_client
    if _async_client:
        await _async_client.aclose()
        _async_client = None


# ── JSON-aware cache helpers ──────────────────────────────────────────────────

async def cache_get(key: str) -> Optional[Any]:
    """
    Retrieves a value from Redis cache. Returns deserialized Python object or None.
    """
    try:
        r = await get_redis_async()
        raw = await r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"cache_get failed key={key}: {e}")
        return None


async def cache_set(key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    """
    Stores a value in Redis cache with optional TTL. Returns True on success.
    """
    try:
        settings = get_settings()
        ttl = ttl_seconds or settings.REDIS_CACHE_TTL_SECONDS
        r = await get_redis_async()
        await r.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.warning(f"cache_set failed key={key}: {e}")
        return False


def cache_get_sync(key: str) -> Optional[Any]:
    try:
        r = get_redis_sync()
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"cache_get_sync failed key={key}: {e}")
        return None


def cache_set_sync(key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
    try:
        settings = get_settings()
        ttl = ttl_seconds or settings.REDIS_CACHE_TTL_SECONDS
        r = get_redis_sync()
        r.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception as e:
        logger.warning(f"cache_set_sync failed key={key}: {e}")
        return False


# ── Pub/Sub Publisher (Safety Alerts) ────────────────────────────────────────

ALERTS_CHANNEL = "bluefish:safety_alerts"


def publish_alert_sync(alert: dict) -> None:
    """
    Publishes a safety alert to the Redis Pub/Sub channel.
    Called by the Fleet Command Agent background loop.
    """
    try:
        r = get_redis_sync()
        r.publish(ALERTS_CHANNEL, json.dumps(alert, default=str))
        logger.info(f"Alert published: {alert.get('type', 'unknown')} mmsi={alert.get('vessel_1_mmsi', alert.get('mmsi', 'N/A'))}")
    except Exception as e:
        logger.error(f"Failed to publish alert: {e}")


# ── Geo-Index Helpers (Live Vessel Tracking) ──────────────────────────────────

def update_vessel_position_sync(mmsi: str, lat: float, lon: float, meta: dict) -> None:
    """
    Updates a vessel's position in the Redis Geo index and companion hash.
    Called by the telemetry ingestion endpoint.

    Schema:
      GEOADD fleet:live <lon> <lat> <mmsi>
      SET fleet:meta:<mmsi> <json_meta> EX 300  (5 min TTL - stale = offline)
    """
    try:
        settings = get_settings()
        r = get_redis_sync()
        pipe = r.pipeline()
        pipe.geoadd(settings.REDIS_GEO_KEY, [lon, lat, mmsi])
        pipe.setex(
            f"{settings.REDIS_META_PREFIX}{mmsi}",
            300,  # 5 min TTL before considered offline
            json.dumps(meta, default=str),
        )
        pipe.execute()
    except Exception as e:
        logger.error(f"Failed to update vessel position mmsi={mmsi}: {e}")


async def check_redis_health() -> bool:
    """Pings Redis. Used in /health endpoint."""
    try:
        r = await get_redis_async()
        return await r.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False
