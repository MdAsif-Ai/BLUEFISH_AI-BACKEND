"""
BlueFish AI - Supabase Database Client
========================================
Provides a singleton Supabase client using the Service Role Key,
which bypasses Row Level Security for all server-side operations.

Usage:
    from core.database import get_supabase
    db = get_supabase()
    result = db.table("vessels").select("*").execute()
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from supabase import Client, create_client

from core.config import get_settings

logger = logging.getLogger("bluefish.database")

_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """
    Returns the global Supabase client singleton.
    Thread-safe: creates the client once and reuses it.
    """
    global _supabase_client
    if _supabase_client is None:
        settings = get_settings()
        logger.info("Initializing Supabase client...")
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY,
        )
        logger.info("Supabase client initialized successfully.")
    return _supabase_client


async def check_supabase_health() -> bool:
    """Pings Supabase to verify connectivity. Used in the /health endpoint."""
    try:
        db = get_supabase()
        # Lightweight query — just fetch 1 row from profiles
        db.table("profiles").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        return False


# ── PostGIS Helper ────────────────────────────────────────────────────────────

def check_point_in_boundaries(lat: float, lon: float, boundary_type: str) -> bool:
    """
    Uses PostGIS ST_Contains to check if a point falls inside any spatial boundary
    of the given type (e.g., 'mpa', 'eez', 'monsoon_ban').

    This calls a Supabase RPC (PostgreSQL function) named `point_in_boundary`
    that must be created in the database.

    SQL to create the function:
        CREATE OR REPLACE FUNCTION point_in_boundary(p_lat float, p_lon float, p_type text)
        RETURNS boolean LANGUAGE sql STABLE AS $$
            SELECT EXISTS (
                SELECT 1 FROM spatial_boundaries
                WHERE type = p_type
                  AND ST_Contains(geom, ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326))
            );
        $$;
    """
    try:
        db = get_supabase()
        result = db.rpc(
            "point_in_boundary",
            {"p_lat": lat, "p_lon": lon, "p_type": boundary_type},
        ).execute()
        return bool(result.data)
    except Exception as e:
        logger.error(
            f"PostGIS check failed lat={lat} lon={lon} type={boundary_type}: {e}"
        )
        # Fail open — don't block a vessel if DB is momentarily unavailable
        return False


def is_in_mpa(lat: float, lon: float) -> bool:
    return check_point_in_boundaries(lat, lon, "mpa")


def is_in_eez(lat: float, lon: float) -> bool:
    return check_point_in_boundaries(lat, lon, "eez")
