"""Authentication service with password hashing, JWT tokens, and OAuth support."""

import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from uuid import UUID

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status

from clients.database import DatabaseClient
from models.auth import (
    UserCreate, UserLogin, UserOAuthCreate, UserResponse,
    AuthProvider, PasswordResetRequest, PasswordReset, PasswordChange
)

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings (should be in config/env)
SECRET_KEY = "your-secret-key-change-this-in-production"  # TODO: Move to env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class AuthService:
    """Authentication service."""
    
    def __init__(self, db: DatabaseClient):
        self.db = db
    
    # Password Hashing
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    # JWT Token Management
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Create refresh token."""
        data = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "jti": secrets.token_urlsafe(32)
        }
        return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.error(f"Token decode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
    
    # User Management
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        query = "SELECT * FROM users WHERE email = %s LIMIT 1"
        result = self.db.execute_query(query, (email,))
        return result[0] if result else None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        query = "SELECT * FROM users WHERE id = %s LIMIT 1"
        result = self.db.execute_query(query, (user_id,))
        return result[0] if result else None
    
    def create_manual_user(self, user_data: UserCreate) -> Dict[str, Any]:
        """Create a new user with manual authentication."""
        # Check if user exists
        existing = self.get_user_by_email(user_data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        password_hash = self.hash_password(user_data.password)
        
        # Generate verification token
        verification_token = secrets.token_urlsafe(32)
        verification_expires = datetime.utcnow() + timedelta(hours=24)
        
        # Insert user
        query = """
            INSERT INTO users (
                name, email, password_hash, auth_provider,
                verification_token, verification_token_expires_at,
                created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """
        
        now = datetime.utcnow()
        result = self.db.execute_query(
            query,
            (
                user_data.name,
                user_data.email,
                password_hash,
                AuthProvider.MANUAL.value,
                verification_token,
                verification_expires,
                now,
                now
            )
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        user = result[0]
        logger.info(f"Created manual user: {user['email']}")
        
        # TODO: Send verification email
        # self.send_verification_email(user['email'], verification_token)
        
        return user
    
    def create_oauth_user(self, user_data: UserOAuthCreate) -> Dict[str, Any]:
        """Create or update user from OAuth provider."""
        # Check if user exists
        existing = self.get_user_by_email(user_data.email)
        
        if existing:
            # Update OAuth info if user exists
            query = """
                UPDATE users
                SET oauth_provider_id = %s,
                    auth_provider = %s,
                    avatar_url = %s,
                    is_verified = true,
                    email_verified_at = %s,
                    updated_at = %s
                WHERE email = %s
                RETURNING *
            """
            result = self.db.execute_query(
                query,
                (
                    user_data.oauth_provider_id,
                    user_data.auth_provider.value,
                    user_data.avatar_url,
                    datetime.utcnow(),
                    datetime.utcnow(),
                    user_data.email
                )
            )
        else:
            # Create new OAuth user
            query = """
                INSERT INTO users (
                    name, email, auth_provider, oauth_provider_id,
                    avatar_url, is_verified, email_verified_at,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """
            now = datetime.utcnow()
            result = self.db.execute_query(
                query,
                (
                    user_data.name,
                    user_data.email,
                    user_data.auth_provider.value,
                    user_data.oauth_provider_id,
                    user_data.avatar_url,
                    True,
                    now,
                    now,
                    now
                )
            )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create/update OAuth user"
            )
        
        user = result[0]
        logger.info(f"Created/updated OAuth user: {user['email']} via {user_data.auth_provider}")
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password."""
        user = self.get_user_by_email(email)
        
        if not user:
            # Log failed attempt
            self.log_login_attempt(email, False, "User not found")
            return None
        
        if user['auth_provider'] != AuthProvider.MANUAL.value:
            # Log failed attempt
            self.log_login_attempt(email, False, f"User registered via {user['auth_provider']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This account uses {user['auth_provider']} authentication"
            )
        
        if not self.verify_password(password, user['password_hash']):
            # Log failed attempt
            self.log_login_attempt(email, False, "Invalid password")
            return None
        
        # Log successful attempt
        self.log_login_attempt(email, True)
        
        # Update last login
        self.update_last_login(user['id'])
        
        return user
    
    def update_last_login(self, user_id: str):
        """Update user's last login timestamp."""
        query = "UPDATE users SET last_login_at = %s WHERE id = %s"
        self.db.execute_query(query, (datetime.utcnow(), user_id), fetch=False)
    
    def log_login_attempt(self, email: str, success: bool, failure_reason: Optional[str] = None):
        """Log login attempt for security monitoring."""
        query = """
            INSERT INTO login_attempts (email, success, failure_reason, attempted_at)
            VALUES (%s, %s, %s, %s)
        """
        self.db.execute_query(
            query,
            (email, success, failure_reason, datetime.utcnow()),
            fetch=False
        )
    
    # Password Reset
    
    def request_password_reset(self, email: str) -> bool:
        """Request password reset."""
        user = self.get_user_by_email(email)
        
        if not user:
            # Don't reveal if email exists
            return True
        
        if user['auth_provider'] != AuthProvider.MANUAL.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This account uses {user['auth_provider']} authentication"
            )
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        reset_expires = datetime.utcnow() + timedelta(hours=1)
        
        # Save token
        query = """
            UPDATE users
            SET reset_token = %s,
                reset_token_expires_at = %s,
                updated_at = %s
            WHERE id = %s
        """
        self.db.execute_query(
            query,
            (reset_token, reset_expires, datetime.utcnow(), user['id']),
            fetch=False
        )
        
        logger.info(f"Password reset requested for: {email}")
        
        # TODO: Send reset email
        # self.send_password_reset_email(email, reset_token)
        
        return True
    
    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password with token."""
        # Find user by token
        query = """
            SELECT * FROM users
            WHERE reset_token = %s
            AND reset_token_expires_at > %s
            LIMIT 1
        """
        result = self.db.execute_query(query, (token, datetime.utcnow()))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        user = result[0]
        
        # Hash new password
        password_hash = self.hash_password(new_password)
        
        # Update password and clear token
        query = """
            UPDATE users
            SET password_hash = %s,
                reset_token = NULL,
                reset_token_expires_at = NULL,
                updated_at = %s
            WHERE id = %s
        """
        self.db.execute_query(
            query,
            (password_hash, datetime.utcnow(), user['id']),
            fetch=False
        )
        
        logger.info(f"Password reset completed for: {user['email']}")
        return True
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change password for logged-in user."""
        user = self.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if user['auth_provider'] != AuthProvider.MANUAL.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"This account uses {user['auth_provider']} authentication"
            )
        
        # Verify current password
        if not self.verify_password(current_password, user['password_hash']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        password_hash = self.hash_password(new_password)
        
        # Update password
        query = """
            UPDATE users
            SET password_hash = %s,
                updated_at = %s
            WHERE id = %s
        """
        self.db.execute_query(
            query,
            (password_hash, datetime.utcnow(), user_id),
            fetch=False
        )
        
        logger.info(f"Password changed for user: {user['email']}")
        return True
    
    # Email Verification
    
    def verify_email(self, token: str) -> bool:
        """Verify email with token."""
        query = """
            SELECT * FROM users
            WHERE verification_token = %s
            AND verification_token_expires_at > %s
            LIMIT 1
        """
        result = self.db.execute_query(query, (token, datetime.utcnow()))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        user = result[0]
        
        # Mark as verified
        query = """
            UPDATE users
            SET is_verified = true,
                email_verified_at = %s,
                verification_token = NULL,
                verification_token_expires_at = NULL,
                updated_at = %s
            WHERE id = %s
        """
        self.db.execute_query(
            query,
            (datetime.utcnow(), datetime.utcnow(), user['id']),
            fetch=False
        )
        
        logger.info(f"Email verified for: {user['email']}")
        return True
    
    # Token Generation
    
    def generate_tokens(self, user: Dict[str, Any]) -> Tuple[str, str]:
        """Generate access and refresh tokens for user."""
        access_token = self.create_access_token(
            data={"sub": str(user['id']), "email": user['email']}
        )
        refresh_token = self.create_refresh_token(str(user['id']))
        
        # Store session
        self.create_session(user['id'], access_token, refresh_token)
        
        return access_token, refresh_token
    
    def create_session(self, user_id: str, access_token: str, refresh_token: str):
        """Create user session."""
        # Decode to get JTI
        payload = self.decode_token(access_token)
        jti = payload.get('jti')
        expires_at = datetime.fromtimestamp(payload.get('exp'))
        
        query = """
            INSERT INTO user_sessions (user_id, token_jti, refresh_token, expires_at)
            VALUES (%s, %s, %s, %s)
        """
        self.db.execute_query(
            query,
            (user_id, jti, refresh_token, expires_at),
            fetch=False
        )
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[str, str]:
        """Refresh access token using refresh token."""
        try:
            payload = self.decode_token(refresh_token)
            
            if payload.get('type') != 'refresh':
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            user_id = payload.get('sub')
            user = self.get_user_by_id(user_id)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Generate new tokens
            return self.generate_tokens(user)
            
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
