"""
BlueFish AI — Auth Routes (Production Only)
===========================================
POST /api/v1/auth/register  → Supabase Auth Admin + profiles INSERT
POST /api/v1/auth/login     → Supabase Auth sign_in_with_password
GET  /api/v1/auth/me        → Authenticated user profile & vessels
POST /api/v1/auth/refresh   → Supabase session refresh
POST /api/v1/auth/logout    → Supabase sign_out
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from auth.schemas import (LoginRequest, LoginResponse, RegisterRequest,
                          TokenRefreshRequest, UserProfileResponse)
from auth.service import (get_user_profile, get_user_vessels, login_user,
                          refresh_user_token, register_user)
from core.security import AuthenticatedUser, get_current_user

logger = logging.getLogger("bluefish.auth.routes")
router = APIRouter(prefix="/api/v1/auth", tags=["🔐 Authentication"])


# ── Register ──────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user (fisherman or government)",
)
async def register(payload: RegisterRequest):
    """
    Registers a new user directly in Supabase Auth (admin) and stores profile in PostgreSQL `profiles` table.
    """
    return register_user(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        role=payload.role,
        phone=payload.phone,
        preferred_language=payload.preferred_language,
    )


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in and receive a JWT access_token",
)
async def login(payload: LoginRequest):
    """
    Authenticates against Supabase Auth and returns JWT token + profile info.
    """
    r = login_user(email=payload.email, password=payload.password)
    p = r["profile"]

    return LoginResponse(
        access_token=r["access_token"],
        refresh_token=r["refresh_token"],
        token_type="bearer",
        expires_in=r["expires_in"],
        profile=UserProfileResponse(
            user_id=p["user_id"],
            email=p.get("email"),
            full_name=p.get("full_name"),
            role=p.get("role", "fisherman"),
            phone=p.get("phone"),
            preferred_language=p.get("preferred_language", "en"),
        ),
    )


# ── /me ───────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=Dict[str, Any],
    summary="Get current user profile and vessels",
)
async def get_me(user: AuthenticatedUser = Depends(get_current_user)):
    """
    Retrieves authenticated user details, profile record from Supabase, and registered vessels.
    """
    profile = get_user_profile(user.user_id)
    vessels = get_user_vessels(user.user_id)

    active_alerts = []
    if vessels:
        mmsi_list = [v["mmsi"] for v in vessels if "mmsi" in v]
        if mmsi_list:
            from core.database import get_supabase
            try:
                r = (
                    get_supabase()
                    .table("safety_alerts")
                    .select("id, alert_type, severity, status, created_at")
                    .in_("mmsi", mmsi_list)
                    .eq("status", "active")
                    .limit(10)
                    .execute()
                )
                active_alerts = r.data or []
            except Exception as exc:
                logger.warning(f"Safety alerts fetch failed for user_id={user.user_id}: {exc}")

    return {
        "user_id": user.user_id,
        "email": user.email,
        "role": user.role,
        "profile": profile or {
            "user_id": user.user_id,
            "email": user.email,
            "role": user.role,
            "full_name": None,
            "phone": None,
            "preferred_language": "en",
        },
        "vessels": vessels,
        "active_alerts": active_alerts,
    }


# ── Refresh ───────────────────────────────────────────────────────────────────

@router.post("/refresh", summary="Refresh access token")
async def refresh_token(payload: TokenRefreshRequest):
    """
    Refreshes Supabase session token.
    """
    return refresh_user_token(payload.refresh_token)


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/logout", summary="Sign out")
async def logout(user: AuthenticatedUser = Depends(get_current_user)):
    """
    Signs out user session from Supabase Auth.
    """
    from core.database import get_supabase
    try:
        get_supabase().auth.sign_out()
    except Exception as exc:
        logger.warning(f"Supabase sign_out warning for user_id={user.user_id}: {exc}")

    return {"status": "logged_out", "user_id": user.user_id}
