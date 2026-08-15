"""
BlueFish AI - Security: JWT Verification & RBAC
=================================================
The frontend (Next.js / Flutter) authenticates users via Supabase Auth
and sends a Supabase-issued JWT in the `Authorization: Bearer <token>` header.

This module:
  1. Verifies the JWT using the Supabase JWT Secret (HS256).
  2. Extracts the user UUID and fetches their role from the `profiles` table.
  3. Provides two FastAPI dependency functions:
     - `get_current_user`: Any valid logged-in user (Fisherman OR Government)
     - `require_government`: Restricts access to government dashboard routes only.

Why use the JWT secret directly instead of Supabase's verify_jwt()?
  - The supabase-py admin client uses the service role key, not user JWTs.
  - Using python-jose + the JWT secret is the standard, fast, zero-network-call approach.
  - We still hit the DB once to get the role — this is cached per-request in the dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("bluefish.security")

# HTTPBearer extracts the token from the Authorization: Bearer <token> header.
# auto_error=False so we can return a clean 401 instead of FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False)


class AuthenticatedUser:
    """Represents a verified, authenticated user."""

    def __init__(self, user_id: str, role: str, email: Optional[str] = None):
        self.user_id = user_id
        self.role = role
        self.email = email

    @property
    def is_government(self) -> bool:
        return self.role == "government"

    @property
    def is_fisherman(self) -> bool:
        return self.role == "fisherman"

    def __repr__(self) -> str:
        return f"<AuthenticatedUser user_id={self.user_id} role={self.role}>"


def _decode_jwt(token: str) -> Dict[str, Any]:
    """
    Verifies a JWT. 
    In development, decodes locally using PyJWT and MOCK_JWT_SECRET.
    In production, verifies against Supabase Auth API (handles ES256/HS256 transparently).

    Raises:
        HTTPException 401: If token is expired, malformed, or invalid.
    """
    from core.config import get_settings
    settings = get_settings()

    if settings.ENVIRONMENT == "development":
        import jwt
        try:
            payload = jwt.decode(
                token,
                settings.MOCK_JWT_SECRET,
                algorithms=["HS256"]
            )
            return payload
        except jwt.PyJWTError as e:
            logger.warning(f"Mock JWT decode failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── PRODUCTION: Supabase Auth Verification ───────────────────────────
    try:
        from core.database import get_supabase
        db = get_supabase()
        
        # Verify the token by fetching the user. This guarantees the token is valid.
        user_response = db.auth.get_user(token)
        if not user_response.user:
            raise ValueError("No user returned")
            
        return {
            "sub": str(user_response.user.id),
            "email": user_response.user.email
        }
    except Exception as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _get_user_role_from_db(user_id: str) -> str:
    """
    Fetches the user's role from the `profiles` table using the service role client.
    Uses a simple 1-row query — fast and lightweight.

    Falls back to 'fisherman' if profile doesn't exist (safe default — least privilege).
    """
    try:
        from core.database import get_supabase

        db = get_supabase()
        result = (
            db.table("profiles")
            .select("role")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if result.data:
            return result.data.get("role", "fisherman")
        return "fisherman"
    except Exception as e:
        logger.warning(f"Could not fetch role for user {user_id}: {e}. Defaulting to 'fisherman'.")
        return "fisherman"


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthenticatedUser:
    """
    FastAPI dependency: Verifies the Bearer JWT and returns the authenticated user.

    Usage in route:
        @router.get("/mobile/map")
        async def get_map(user: AuthenticatedUser = Depends(get_current_user)):
            ...
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_jwt(credentials.credentials)
    user_id: Optional[str] = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT is missing the 'sub' (user ID) claim.",
        )

    role = _get_user_role_from_db(user_id)
    email = payload.get("email")

    return AuthenticatedUser(user_id=user_id, role=role, email=email)


async def require_government(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    FastAPI dependency: Restricts a route to government users only.

    Usage in route:
        @router.get("/command/fleet/density")
        async def get_density(user: AuthenticatedUser = Depends(require_government)):
            ...

    Returns 403 Forbidden if the user's role is 'fisherman'.
    """
    if not user.is_government:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires government role access.",
        )
    return user
