"""Authentication models and schemas."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from enum import Enum


class AuthProvider(str, Enum):
    """Authentication provider types."""
    MANUAL = "manual"
    GOOGLE = "google"
    GITHUB = "github"


class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=255)


class UserCreate(UserBase):
    """User creation model for manual signup."""
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str


class UserOAuthCreate(BaseModel):
    """User creation via OAuth."""
    email: EmailStr
    name: str
    auth_provider: AuthProvider
    oauth_provider_id: str
    avatar_url: Optional[str] = None


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    name: str
    auth_provider: str
    is_verified: bool
    avatar_url: Optional[str] = None
    preferred_language: str = "en"
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordReset(BaseModel):
    """Password reset with token."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class PasswordChange(BaseModel):
    """Password change for logged-in users."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class EmailVerification(BaseModel):
    """Email verification model."""
    token: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str
