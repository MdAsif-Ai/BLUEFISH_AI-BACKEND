"""
BlueFish AI - Security: JWT Verification & RBAC (Production Only)
==================================================================
Verifies the Supabase JWT by calling supabase.auth.get_user(token).
Fetches the user's role directly from the Supabase `profiles` table.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("bluefish.security")
_bearer = HTTPBearer(auto_error=False)


class AuthenticatedUser:
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


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthenticatedUser:
    """
    Verifies the Bearer JWT against Supabase Auth and fetches the user's role from PostgreSQL.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        from core.database import get_supabase
        db = get_supabase()

        # 1. Verify token with Supabase Auth
        resp = db.auth.get_user(token)
        if not resp or not resp.user:
            raise ValueError("Supabase did not return a user for this token.")

        user_id = str(resp.user.id)
        email = resp.user.email

        # 2. Fetch role from profiles table
        try:
            profile_resp = db.table("profiles").select("role").eq("id", user_id).single().execute()
            if not profile_resp.data:
                # Failsafe: create default profile if missing
                db.table("profiles").upsert({"id": user_id, "role": "fisherman"}).execute()
                role = "fisherman"
            else:
                role = profile_resp.data.get("role", "fisherman")
        except Exception as p_err:
            logger.warning(f"Profile role fetch failed for user_id={user_id}: {p_err}. Defaulting to 'fisherman'.")
            role = "fisherman"

        return AuthenticatedUser(user_id=user_id, role=role, email=email)

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"Authentication failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_government(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    Restricts a route to government users only.
    """
    if not user.is_government:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Government role required.",
        )
    return user
