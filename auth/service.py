"""
BlueFish AI — Auth Service (Production Only)
============================================
Strict Supabase Auth & PostgreSQL Integration.

• Registration: Uses `supabase.auth.admin.create_user()` to create user in `auth.users`,
  then immediately inserts user record into `profiles` table.
• Login: Uses `supabase.auth.sign_in_with_password()`, then fetches profile details.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

logger = logging.getLogger("bluefish.auth.service")

ALLOWED_ROLES = {"fisherman", "government"}


def register_user(
    email: str,
    password: str,
    full_name: str,
    role: str,
    phone: Optional[str] = None,
    preferred_language: str = "en",
) -> Dict[str, Any]:
    """
    Registers a user in Supabase Auth (admin) and saves profile into PostgreSQL `profiles` table.
    """
    if role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Role must be one of: {ALLOWED_ROLES}",
        )

    from core.database import get_supabase
    db = get_supabase()

    # Step 1: Create user via Supabase Auth Admin API
    try:
        resp = db.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {
                "full_name": full_name,
                "role": role,
            },
        })
    except Exception as exc:
        _raise_auth_error(exc, "Registration")

    user = getattr(resp, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase Auth returned no user object upon creation.",
        )

    user_id = str(user.id)
    logger.info(f"Supabase Auth user created successfully: {user_id} ({email})")

    # Step 2: Insert into PostgreSQL profiles table
    try:
        profile_data = {
            "id": user_id,
            "role": role,
            "full_name": full_name,
            "phone": phone,
            "preferred_language": preferred_language,
        }
        result = db.table("profiles").insert(profile_data).execute()
    except Exception as exc:
        logger.error(f"CRITICAL: Failed to insert profile into DB for user_id={user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Auth user created ({user_id}), but saving profile to database failed: {exc}",
        )

    if not result.data:
        # Check if row exists via fallback check
        try:
            check_p = db.table("profiles").select("id").eq("id", user_id).execute()
            if not check_p.data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Profile insert returned no data. Check database RLS policies or triggers.",
                )
        except HTTPException:
            raise
        except Exception:
            pass

    logger.info(f"Profile saved successfully in profiles table for {user_id}")

    return {
        "status": "registered",
        "user_id": user_id,
        "email": email,
        "full_name": full_name,
        "role": role,
        "message": "Registration successful.",
    }


def login_user(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticates user with Supabase Auth and fetches associated profile.
    """
    from core.database import get_supabase
    db = get_supabase()

    try:
        resp = db.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as exc:
        _raise_auth_error(exc, "Login")

    session, user = getattr(resp, "session", None), getattr(resp, "user", None)
    if not session or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. Invalid session returned.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = str(user.id)

    # Fetch profile details
    profile_row: Dict[str, Any] = {}
    try:
        r = db.table("profiles").select("*").eq("id", user_id).single().execute()
        profile_row = r.data or {}
    except Exception as exc:
        logger.warning(f"Profile fetch failed for user_id={user_id}: {exc}")

    if not profile_row:
        # Auto-create fallback profile if missing
        try:
            r = db.table("profiles").upsert({"id": user_id, "role": "fisherman"}).execute()
            profile_row = r.data[0] if r.data else {"id": user_id, "role": "fisherman"}
        except Exception as exc:
            logger.error(f"Auto-profile fallback creation failed for {user_id}: {exc}")
            profile_row = {"id": user_id, "role": "fisherman"}

    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "expires_in": getattr(session, "expires_in", 3600) or 3600,
        "profile": {
            "user_id": user_id,
            "email": user.email,
            "full_name": profile_row.get("full_name"),
            "role": profile_row.get("role", "fisherman"),
            "phone": profile_row.get("phone"),
            "preferred_language": profile_row.get("preferred_language", "en"),
        },
    }


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetches user profile row from Supabase database."""
    from core.database import get_supabase
    db = get_supabase()
    try:
        r = db.table("profiles").select("*").eq("id", user_id).single().execute()
        return r.data
    except Exception as exc:
        logger.warning(f"Profile fetch failed for user_id={user_id}: {exc}")
        return None


def get_user_vessels(user_id: str) -> list:
    """Fetches vessels associated with user from Supabase database."""
    from core.database import get_supabase
    db = get_supabase()
    try:
        r = db.table("vessels").select("mmsi, vessel_type, vessel_name").eq("owner_id", user_id).execute()
        return r.data or []
    except Exception as exc:
        logger.warning(f"Vessels fetch failed for user_id={user_id}: {exc}")
        return []


def refresh_user_token(refresh_token: str) -> Dict[str, Any]:
    """Refreshes Supabase session token."""
    from core.database import get_supabase
    db = get_supabase()
    try:
        sr = db.auth.refresh_session(refresh_token)
        s = sr.session
        if not s:
            raise ValueError("No session returned")
        return {
            "access_token": s.access_token,
            "refresh_token": s.refresh_token,
            "expires_in": getattr(s, "expires_in", 3600) or 3600,
            "token_type": "bearer",
        }
    except Exception as exc:
        logger.warning(f"Token refresh failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _raise_auth_error(exc: Exception, operation: str) -> None:
    """Maps Supabase Auth exceptions to explicit HTTPExceptions."""
    logger.error(f"Supabase Auth error during {operation}: {type(exc).__name__}: {exc}")
    try:
        from gotrue.errors import AuthApiError, AuthWeakPasswordError
        if isinstance(exc, AuthWeakPasswordError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password is too weak.")
        if isinstance(exc, AuthApiError):
            s = getattr(exc, "status", 400)
            msg = getattr(exc, "message", str(exc))
            if s == 429:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many auth requests. Try again later.")
            if s in (409, 422) or "already registered" in msg.lower() or "already been registered" in msg.lower():
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")
            if s == 400:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.", headers={"WWW-Authenticate": "Bearer"})
            raise HTTPException(status_code=s, detail=msg)
    except HTTPException:
        raise
    except ImportError:
        pass

    err = str(exc).lower()
    if "rate limit" in err or "too many" in err:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests. Please try again later.")
    if "already registered" in err or "already exists" in err or "duplicate" in err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{operation} failed: {exc}")
