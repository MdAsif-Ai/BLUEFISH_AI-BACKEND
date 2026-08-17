"""BlueFish AI — Auth Schemas"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    role: str = Field(..., description="'fisherman' or 'government'")
    phone: Optional[str] = None
    preferred_language: str = "en"

    model_config = {"json_schema_extra": {"example": {
        "email": "rajan@example.com", "password": "securePass123",
        "full_name": "Rajan Kumar", "role": "fisherman",
        "phone": "+919876543210", "preferred_language": "ta",
    }}}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    model_config = {"json_schema_extra": {"example": {
        "email": "rajan@example.com", "password": "securePass123",
    }}}


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
