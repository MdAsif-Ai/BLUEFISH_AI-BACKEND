"""
BlueFish AI - Authentication Routes
=====================================
Handles user registration, login, and profile retrieval for both
the Flutter fisherman app and the Next.js government command center.

All auth operations go through Supabase Auth (supabase-py v2 client-side calls),
and the backend inserts the corresponding `profiles` row using the Service Role Key.

Endpoints:
  POST /api/v1/auth/register  → Create Supabase Auth user + insert profile
  POST /api/v1/auth/login     → Sign in → return JWT access_token + profile
  GET  /api/v1/auth/me        → Return current user's profile + vessel info
  POST /api/v1/auth/logout    → Invalidate refresh token (client-side JWT drop)
  POST /api/v1/auth/refresh   → Exchange refresh_token for new access_token

Design notes:
  - Registration uses the SERVICE ROLE KEY to insert the `profiles` row.
    supabase.auth.sign_up() creates the Supabase Auth user first; on success,
    we get the new user's UUID and insert into `profiles` with role + full_name.
  - Login returns both the raw Supabase session (access_token, refresh_token)
    and the user's `profiles` row so the frontend has everything it needs in
    one round-trip.
  - The `GET /me` endpoint verifies the JWT (via get_current_user dependency)
    and fetches the profile + vessels from Supabase.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from core.security import AuthenticatedUser, get_current_user

logger = logging.getLogger("bluefish.routes.auth")

router = APIRouter(prefix="/api/v1/auth", tags=["🔐 Authentication"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="Minimum 6 characters")
    full_name: str = Field(..., min_length=2)
    role: str = Field(..., description="'fisherman' or 'government'")
    phone: Optional[str] = Field(default=None, description="E.164 format recommended: +919XXXXXXXXX")
    preferred_language: str = Field(default="en", description="ISO 639-1 code: 'en', 'ta', 'hi'")

    model_config = {"json_schema_extra": {
        "example": {
            "email": "rajan@example.com",
            "password": "securePass123",
            "full_name": "Rajan Kumar",
            "role": "fisherman",
            "phone": "+919876543210",
            "preferred_language": "ta",
        }
    }}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class UserProfileResponse(BaseModel):
    user_id: str
    email: Optional[str]
    full_name: Optional[str]
    role: str
    phone: Optional[str]
    preferred_language: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    profile: UserProfileResponse


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_profile_response(user_id: str, email: Optional[str], profile_row: Dict[str, Any]) -> UserProfileResponse:
    return UserProfileResponse(
        user_id=user_id,
        email=email,
        full_name=profile_row.get("full_name"),
        role=profile_row.get("role", "fisherman"),
        phone=profile_row.get("phone"),
        preferred_language=profile_row.get("preferred_language", "en"),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user (fisherman or government)",
)
async def register(payload: RegisterRequest):
    """
    Registers a new user with two steps:
    1. Creates the Supabase Auth user via `auth.sign_up()`.
    2. Inserts a row into the `profiles` table using the Service Role Key.

    Both steps must succeed. If profile insertion fails after auth creation,
    we log the error but still return success (the user can login; profile is
    populated on first `GET /me` call via upsert).

    Returns the user's UUID on success. The user must then `POST /login`
    to obtain their JWT access_token.
    """
    from core.database import get_supabase

    # Validate role
    allowed_roles = {"fisherman", "government"}
    if payload.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"role must be one of {allowed_roles}",
        )

    from core.config import get_settings
    settings = get_settings()
    db = get_supabase()

    if settings.ENVIRONMENT == "development":
        import uuid
        import time
        import jwt
        
        # Deterministic UUID based on email so login can find it
        user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, payload.email))
        
        try:
            db.table("profiles").upsert({
                "id": user_id,
                "role": payload.role,
                "full_name": payload.full_name,
                "phone": payload.phone,
                "preferred_language": payload.preferred_language,
            }).execute()
            logger.info(f"[DEV MODE] Profile mocked for {user_id} role={payload.role}")
        except Exception as e:
            err_str = str(e).lower()
            if "duplicate key" in err_str or "unique constraint" in err_str:
                raise HTTPException(status_code=409, detail="User already registered.")
            elif "foreign key constraint" in err_str:
                logger.warning(f"[DEV MODE] Skipping profile insert due to auth.users FK constraint: {e}")
            else:
                raise HTTPException(status_code=400, detail=f"Mock registration failed: {e}")

        return {
            "status": "registered",
            "user_id": user_id,
            "email": payload.email,
            "message": "Registration successful (Mock Auth).",
        }

    # ── PRODUCTION: Supabase Auth ────────────────────────────────────
    try:
        sign_up_response = db.auth.sign_up({
            "email": payload.email,
            "password": payload.password,
        })
    except Exception as e:
        logger.error(f"Supabase auth sign_up failed: {e}")
        from gotrue.errors import AuthApiError, AuthWeakPasswordError
        if isinstance(e, AuthApiError):
            status_code = getattr(e, "status", 400)
            msg = getattr(e, "message", str(e))
            if status_code == 429:
                raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
            elif status_code == 409 or "already registered" in msg.lower():
                raise HTTPException(status_code=409, detail="User already registered.")
            elif status_code == 400 and "weak" in msg.lower():
                raise HTTPException(status_code=400, detail="Password is too weak.")
            else:
                raise HTTPException(status_code=status_code, detail=msg)
        elif isinstance(e, AuthWeakPasswordError):
            raise HTTPException(status_code=400, detail="Password is too weak.")
            
        # Fallback string matching
        error_msg = str(e).lower()
        if "rate limit" in error_msg or "too many requests" in error_msg:
            raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
        elif "already registered" in error_msg:
            raise HTTPException(status_code=409, detail="User already registered.")
        elif "weak" in error_msg:
            raise HTTPException(status_code=400, detail="Password is too weak.")
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {e}",
        )

    user = sign_up_response.user
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed: Supabase did not return a user. Email may already be registered.",
        )

    user_id = str(user.id)

    # ── Step 2: Insert profile row (Service Role Key — bypasses RLS) ─────────
    try:
        db.table("profiles").upsert({
            "id": user_id,
            "role": payload.role,
            "full_name": payload.full_name,
            "phone": payload.phone,
            "preferred_language": payload.preferred_language,
        }).execute()
        logger.info(f"Profile created for user {user_id} role={payload.role}")
    except Exception as e:
        # Non-fatal: Auth user was created. Profile will be created on first login.
        logger.error(f"Profile insert failed for {user_id}: {e}")

    return {
        "status": "registered",
        "user_id": user_id,
        "email": payload.email,
        "message": (
            "Registration successful. "
            "If email confirmation is enabled on your Supabase project, "
            "please check your inbox before logging in."
        ),
    }


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Sign in and get JWT access_token",
)
async def login(payload: LoginRequest):
    """
    Authenticates the user via Supabase Auth and returns:
    - `access_token`: Short-lived JWT — send as `Authorization: Bearer <token>` header
    - `refresh_token`: Long-lived token for silent re-authentication
    - `profile`: Full user profile from the `profiles` table

    The `access_token` is a Supabase-issued HS256 JWT. It is verified on
    subsequent requests by `core/security.py::get_current_user()`.
    """
    from core.database import get_supabase
    from core.config import get_settings

    settings = get_settings()
    db = get_supabase()

    if settings.ENVIRONMENT == "development":
        import uuid
        import time
        import jwt
        
        user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, payload.email))
        
        # Fetch profile
        try:
            profile_result = db.table("profiles").select("*").eq("id", user_id).single().execute()
            profile_row = profile_result.data or {}
        except Exception:
            profile_row = {}
            
        if not profile_row:
            logger.warning(f"[DEV MODE] Profile missing for {user_id}. Using mock data.")
            profile_row = {
                "id": user_id,
                "role": "fisherman",
                "full_name": "Mock User",
                "phone": "+910000000000",
                "preferred_language": "en"
            }

        # Generate mock JWT
        expires_in = 3600
        jwt_payload = {
            "sub": user_id,
            "email": payload.email,
            "role": profile_row.get("role", "fisherman"),
            "exp": int(time.time()) + expires_in,
            "iat": int(time.time()),
        }
        access_token = jwt.encode(jwt_payload, settings.MOCK_JWT_SECRET, algorithm="HS256")
        
        profile = _build_profile_response(user_id, payload.email, profile_row)
        return LoginResponse(
            access_token=access_token,
            refresh_token="mock_refresh_token",
            token_type="bearer",
            expires_in=expires_in,
            profile=profile,
        )

    # ── PRODUCTION: Supabase Auth ────────────────────────────────────────────
    try:
        sign_in_response = db.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password,
        })
    except Exception as e:
        logger.warning(f"Login failed for {payload.email}: {e}")
        from gotrue.errors import AuthApiError
        if isinstance(e, AuthApiError):
            status_code = getattr(e, "status", 401)
            msg = getattr(e, "message", str(e))
            if status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            elif status_code == 400:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                raise HTTPException(status_code=status_code, detail=msg, headers={"WWW-Authenticate": "Bearer"})

        error_msg = str(e).lower()
        if "rate limit" in error_msg or "too many requests" in error_msg:
             raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    session = sign_in_response.session
    user = sign_in_response.user

    if session is None or user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed — no session returned.",
        )

    user_id = str(user.id)
    email = user.email

    # Fetch profile from `profiles` table
    profile_row: Dict[str, Any] = {}
    try:
        profile_result = (
            db.table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )
        profile_row = profile_result.data or {}
    except Exception as e:
        logger.warning(f"Profile fetch failed on login for {user_id}: {e}. Using defaults.")

    # If profile row doesn't exist yet (e.g., first login after manual Supabase signup),
    # create it with minimal data
    if not profile_row:
        try:
            db.table("profiles").upsert({"id": user_id, "role": "fisherman"}).execute()
            profile_row = {"id": user_id, "role": "fisherman"}
        except Exception:
            profile_row = {"role": "fisherman"}

    profile = _build_profile_response(user_id, email, profile_row)

    return LoginResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        token_type="bearer",
        expires_in=session.expires_in or 3600,
        profile=profile,
    )


@router.get(
    "/me",
    response_model=Dict[str, Any],
    summary="Get current user profile and vessels",
)
async def get_me(user: AuthenticatedUser = Depends(get_current_user)):
    """
    Returns the authenticated user's:
    - Profile (from `profiles` table)
    - Registered vessels (from `vessels` table)
    - Active safety alerts for their vessels

    Requires `Authorization: Bearer <access_token>` header.
    """
    from core.database import get_supabase

    db = get_supabase()
    result: Dict[str, Any] = {
        "user_id": user.user_id,
        "email": user.email,
        "role": user.role,
        "profile": None,
        "vessels": [],
        "active_alerts": [],
    }

    # ── Profile ───────────────────────────────────────────────────────────────
    try:
        profile_result = (
            db.table("profiles")
            .select("*")
            .eq("id", user.user_id)
            .single()
            .execute()
        )
        result["profile"] = profile_result.data
    except Exception as e:
        logger.warning(f"Profile fetch failed for {user.user_id}: {e}")
        
    if not result["profile"]:
        from core.config import get_settings
        if get_settings().ENVIRONMENT == "development":
            result["profile"] = {
                "id": user.user_id,
                "role": user.role,
                "full_name": "Mock User",
                "phone": "+910000000000",
                "preferred_language": "en"
            }

    # ── Vessels ───────────────────────────────────────────────────────────────
    try:
        vessels_result = (
            db.table("vessels")
            .select("mmsi, vessel_type")
            .eq("owner_id", user.user_id)
            .execute()
        )
        result["vessels"] = vessels_result.data or []
    except Exception as e:
        logger.warning(f"Vessels fetch failed for {user.user_id}: {e}")

    # ── Active alerts for this user's vessels ─────────────────────────────────
    if result["vessels"]:
        mmsi_list = [v["mmsi"] for v in result["vessels"]]
        try:
            alerts_result = (
                db.table("safety_alerts")
                .select("id, alert_type, severity, status, created_at")
                .in_("mmsi", mmsi_list)
                .eq("status", "active")
                .limit(10)
                .execute()
            )
            result["active_alerts"] = alerts_result.data or []
        except Exception as e:
            logger.warning(f"Active alerts fetch failed: {e}")

    return result


@router.post("/logout", summary="Sign out (invalidate refresh token)")
async def logout(user: AuthenticatedUser = Depends(get_current_user)):
    """
    Signs the user out.
    """
    from core.config import get_settings
    from core.database import get_supabase

    settings = get_settings()
    
    if settings.ENVIRONMENT == "development":
        logger.info(f"[DEV MODE] Logout called for {user.user_id}")
        return {"status": "logged_out", "user_id": user.user_id}

    db = get_supabase()
    try:
        db.auth.sign_out()
    except Exception as e:
        logger.warning(f"Sign-out call failed for {user.user_id}: {e}")

    return {"status": "logged_out", "user_id": user.user_id}


@router.post("/refresh", summary="Refresh access token using refresh_token")
async def refresh_token(payload: TokenRefreshRequest):
    """
    Exchanges a valid `refresh_token` for a new `access_token`.
    """
    from core.config import get_settings
    from core.database import get_supabase

    settings = get_settings()

    if settings.ENVIRONMENT == "development":
        if payload.refresh_token == "mock_refresh_token":
            import time
            import jwt
            # Generate a new mock token
            expires_in = 3600
            jwt_payload = {
                "sub": "mock_user",
                "email": "mock_user@example.com",
                "role": "fisherman",
                "exp": int(time.time()) + expires_in,
                "iat": int(time.time()),
            }
            access_token = jwt.encode(jwt_payload, settings.MOCK_JWT_SECRET, algorithm="HS256")
            return {
                "access_token": access_token,
                "refresh_token": "mock_refresh_token",
                "expires_in": expires_in,
                "token_type": "bearer",
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid mock refresh token.")

    db = get_supabase()
    try:
        session_response = db.auth.refresh_session(payload.refresh_token)
        session = session_response.session
        if session is None:
            raise ValueError("No session returned")
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "expires_in": session.expires_in or 3600,
            "token_type": "bearer",
        }
    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
