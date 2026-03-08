"""
Authentication router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, EmailStr, Field

from services.auth import AuthService, extract_bearer_token, get_auth_service

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class TokenExchangeRequest(BaseModel):
    cognito_id: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    preferred_language: str = Field(default="en", min_length=2, max_length=10)
    tech_stack: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None


@router.post("/token")
async def exchange_token(
    payload: TokenExchangeRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Exchange trusted identity data for a backend JWT.

    Intended for use after an upstream identity provider authenticates the user.
    """
    return auth_service.issue_token(
        cognito_id=payload.cognito_id,
        email=payload.email,
        preferred_language=payload.preferred_language,
        tech_stack=payload.tech_stack,
        preferences=payload.preferences,
    )


@router.get("/me")
async def get_authenticated_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
):
    token = extract_bearer_token(authorization)
    return auth_service.get_current_user(token)
