"""
Authentication service for JWT issuance and user synchronization.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt

from config import settings
from clients.database import DatabaseClient


class AuthService:
    """Issue backend JWTs and keep the users table in sync."""

    def __init__(self, db_client: DatabaseClient):
        self.db = db_client

    def issue_token(
        self,
        cognito_id: str,
        email: str,
        preferred_language: str = "en",
        tech_stack: Optional[List[str]] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not settings.secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SECRET_KEY is not configured",
            )

        user = self._upsert_user(
            cognito_id=cognito_id,
            email=email,
            preferred_language=preferred_language,
            tech_stack=tech_stack or [],
            preferences=preferences or {},
        )

        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_expiration_minutes
        )
        token = jwt.encode(
            {
                "sub": str(user["id"]),
                "email": user["email"],
                "cognito_id": user["cognito_id"],
                "exp": expires_at,
            },
            settings.secret_key,
            algorithm=settings.jwt_algorithm,
        )

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_at": expires_at.isoformat(),
            "user": self._serialize_user(user),
        }

    def get_current_user(self, token: str) -> Dict[str, Any]:
        if not settings.secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SECRET_KEY is not configured",
            )

        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm],
            )
        except JWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            ) from exc

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        rows = self.db.execute_query(
            """
            SELECT id, cognito_id, email, preferred_language, tech_stack, preferences,
                   created_at, last_login
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        )
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated user not found",
            )

        return self._serialize_user(rows[0])

    def _upsert_user(
        self,
        cognito_id: str,
        email: str,
        preferred_language: str,
        tech_stack: List[str],
        preferences: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows = self.db.execute_query(
            """
            INSERT INTO users (
                cognito_id, email, preferred_language, tech_stack, preferences, last_login
            )
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, NOW())
            ON CONFLICT (cognito_id)
            DO UPDATE SET
                email = EXCLUDED.email,
                preferred_language = EXCLUDED.preferred_language,
                tech_stack = EXCLUDED.tech_stack,
                preferences = EXCLUDED.preferences,
                last_login = NOW()
            RETURNING id, cognito_id, email, preferred_language, tech_stack, preferences,
                      created_at, last_login
            """,
            (
                cognito_id,
                email,
                preferred_language,
                self._to_json(tech_stack),
                self._to_json(preferences),
            ),
        )
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to persist authenticated user",
            )
        return rows[0]

    @staticmethod
    def _to_json(value: Any) -> str:
        import json

        return json.dumps(value)

    @staticmethod
    def _serialize_user(user: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(user["id"]),
            "cognito_id": user["cognito_id"],
            "email": user["email"],
            "preferred_language": user.get("preferred_language", "en"),
            "tech_stack": user.get("tech_stack") or [],
            "preferences": user.get("preferences") or {},
            "created_at": user.get("created_at").isoformat() if user.get("created_at") else None,
            "last_login": user.get("last_login").isoformat() if user.get("last_login") else None,
        }


def get_auth_service(req: Request) -> AuthService:
    db_client = getattr(req.app.state, "db", None) or DatabaseClient()
    return AuthService(db_client)


def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer token",
        )
    return token
