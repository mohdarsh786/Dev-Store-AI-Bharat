"""Authentication router with all auth endpoints."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from clients.database import DatabaseClient
from services.auth_service import AuthService
from models.auth import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    PasswordResetRequest, PasswordReset, PasswordChange,
    EmailVerification, RefreshTokenRequest, UserOAuthCreate
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])
security = HTTPBearer()

# Dependency to get auth service
def get_auth_service() -> AuthService:
    """Get authentication service instance."""
    db = DatabaseClient()
    return AuthService(db)

# Dependency to get current user from token
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    try:
        payload = auth_service.decode_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        user = auth_service.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# ============================================================================
# SIGNUP ENDPOINTS
# ============================================================================

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Sign up a new user with email and password.
    
    - **name**: User's full name (min 2 characters)
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, uppercase, lowercase, digit)
    
    Returns access token and user info.
    """
    try:
        # Create user
        user = auth_service.create_manual_user(user_data)
        
        # Generate tokens
        access_token, refresh_token = auth_service.generate_tokens(user)
        
        # Prepare response
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            auth_provider=user['auth_provider'],
            is_verified=user['is_verified'],
            avatar_url=user.get('avatar_url'),
            preferred_language=user.get('preferred_language', 'en'),
            created_at=user['created_at'],
            last_login_at=user.get('last_login_at')
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,  # 30 minutes
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account"
        )


# ============================================================================
# LOGIN ENDPOINTS
# ============================================================================

@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login with email and password.
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns access token and user info.
    """
    try:
        # Authenticate user
        user = auth_service.authenticate_user(credentials.email, credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Generate tokens
        access_token, refresh_token = auth_service.generate_tokens(user)
        
        # Prepare response
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            auth_provider=user['auth_provider'],
            is_verified=user['is_verified'],
            avatar_url=user.get('avatar_url'),
            preferred_language=user.get('preferred_language', 'en'),
            created_at=user['created_at'],
            last_login_at=user.get('last_login_at')
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,  # 30 minutes
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


# ============================================================================
# OAUTH ENDPOINTS
# ============================================================================

@router.post("/oauth/google", response_model=TokenResponse)
async def google_oauth(
    user_data: UserOAuthCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login or signup with Google OAuth.
    
    Frontend should handle Google OAuth flow and send user data here.
    """
    try:
        # Create or update user
        user = auth_service.create_oauth_user(user_data)
        
        # Generate tokens
        access_token, refresh_token = auth_service.generate_tokens(user)
        
        # Prepare response
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            auth_provider=user['auth_provider'],
            is_verified=user['is_verified'],
            avatar_url=user.get('avatar_url'),
            preferred_language=user.get('preferred_language', 'en'),
            created_at=user['created_at'],
            last_login_at=user.get('last_login_at')
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,
            user=user_response
        )
        
    except Exception as e:
        logger.error(f"Google OAuth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth authentication failed"
        )


@router.post("/oauth/github", response_model=TokenResponse)
async def github_oauth(
    user_data: UserOAuthCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login or signup with GitHub OAuth.
    
    Frontend should handle GitHub OAuth flow and send user data here.
    """
    try:
        # Create or update user
        user = auth_service.create_oauth_user(user_data)
        
        # Generate tokens
        access_token, refresh_token = auth_service.generate_tokens(user)
        
        # Prepare response
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            auth_provider=user['auth_provider'],
            is_verified=user['is_verified'],
            avatar_url=user.get('avatar_url'),
            preferred_language=user.get('preferred_language', 'en'),
            created_at=user['created_at'],
            last_login_at=user.get('last_login_at')
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,
            user=user_response
        )
        
    except Exception as e:
        logger.error(f"GitHub OAuth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth authentication failed"
        )


# ============================================================================
# PASSWORD MANAGEMENT
# ============================================================================

@router.post("/password/forgot")
async def forgot_password(
    request: PasswordResetRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Request password reset email.
    
    - **email**: User's email address
    
    Always returns success (don't reveal if email exists).
    """
    try:
        auth_service.request_password_reset(request.email)
        return {
            "message": "If the email exists, a password reset link has been sent"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        return {
            "message": "If the email exists, a password reset link has been sent"
        }


@router.post("/password/reset")
async def reset_password(
    reset_data: PasswordReset,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Reset password with token from email.
    
    - **token**: Reset token from email
    - **new_password**: New strong password
    """
    try:
        auth_service.reset_password(reset_data.token, reset_data.new_password)
        return {"message": "Password reset successful"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )


@router.post("/password/change")
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Change password for logged-in user.
    
    Requires authentication.
    
    - **current_password**: Current password
    - **new_password**: New strong password
    """
    try:
        auth_service.change_password(
            str(current_user['id']),
            password_data.current_password,
            password_data.new_password
        )
        return {"message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


# ============================================================================
# EMAIL VERIFICATION
# ============================================================================

@router.post("/verify-email")
async def verify_email(
    verification: EmailVerification,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verify email with token from email.
    
    - **token**: Verification token from email
    """
    try:
        auth_service.verify_email(verification.token)
        return {"message": "Email verified successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification failed"
        )


# ============================================================================
# TOKEN MANAGEMENT
# ============================================================================

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    """
    try:
        access_token, refresh_token = auth_service.refresh_access_token(request.refresh_token)
        
        # Get user info
        payload = auth_service.decode_token(access_token)
        user = auth_service.get_user_by_id(payload['sub'])
        
        user_response = UserResponse(
            id=str(user['id']),
            email=user['email'],
            name=user['name'],
            auth_provider=user['auth_provider'],
            is_verified=user['is_verified'],
            avatar_url=user.get('avatar_url'),
            preferred_language=user.get('preferred_language', 'en'),
            created_at=user['created_at'],
            last_login_at=user.get('last_login_at')
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,
            user=user_response
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


# ============================================================================
# USER INFO
# ============================================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
):
    """
    Get current authenticated user info.
    
    Requires authentication.
    """
    return UserResponse(
        id=str(current_user['id']),
        email=current_user['email'],
        name=current_user['name'],
        auth_provider=current_user['auth_provider'],
        is_verified=current_user['is_verified'],
        avatar_url=current_user.get('avatar_url'),
        preferred_language=current_user.get('preferred_language', 'en'),
        created_at=current_user['created_at'],
        last_login_at=current_user.get('last_login_at')
    )


@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_user)
):
    """
    Logout current user.
    
    Requires authentication.
    """
    # TODO: Invalidate token/session
    return {"message": "Logged out successfully"}
