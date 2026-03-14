-- Migration: Enhanced users table with full authentication support
-- Description: Support for manual auth, OAuth (Google/GitHub), password reset
-- Date: 2024

-- Drop existing users table if needed (for development)
-- DROP TABLE IF EXISTS users CASCADE;

-- Create enhanced users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic Info
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    
    -- Authentication
    password_hash VARCHAR(255),  -- NULL for OAuth users
    auth_provider VARCHAR(50) NOT NULL DEFAULT 'manual',  -- 'manual', 'google', 'github'
    oauth_provider_id VARCHAR(255),  -- OAuth provider user ID
    
    -- Account Status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    email_verified_at TIMESTAMP,
    
    -- Password Reset
    reset_token VARCHAR(255),
    reset_token_expires_at TIMESTAMP,
    
    -- Email Verification
    verification_token VARCHAR(255),
    verification_token_expires_at TIMESTAMP,
    
    -- User Preferences
    preferred_language VARCHAR(10) DEFAULT 'en',
    tech_stack JSONB DEFAULT '[]'::jsonb,
    preferences JSONB DEFAULT '{}'::jsonb,
    
    -- Profile
    avatar_url TEXT,
    bio TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP,
    
    -- Constraints
    CONSTRAINT check_auth_provider CHECK (auth_provider IN ('manual', 'google', 'github')),
    CONSTRAINT check_manual_password CHECK (
        (auth_provider = 'manual' AND password_hash IS NOT NULL) OR
        (auth_provider != 'manual')
    )
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_auth_provider ON users(auth_provider);
CREATE INDEX IF NOT EXISTS idx_users_oauth_provider_id ON users(oauth_provider_id);
CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token);
CREATE INDEX IF NOT EXISTS idx_users_verification_token ON users(verification_token);
CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at DESC);

-- Create sessions table for JWT token management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_jti VARCHAR(255) UNIQUE NOT NULL,  -- JWT ID
    refresh_token VARCHAR(255) UNIQUE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token_jti ON user_sessions(token_jti);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON user_sessions(expires_at);

-- Create login attempts table for security
CREATE TABLE IF NOT EXISTS login_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45),
    success BOOLEAN NOT NULL,
    failure_reason VARCHAR(255),
    attempted_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_login_attempts_email ON login_attempts(email);
CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip_address);
CREATE INDEX IF NOT EXISTS idx_login_attempts_attempted_at ON login_attempts(attempted_at DESC);

-- Add comments
COMMENT ON TABLE users IS 'User accounts with support for manual and OAuth authentication';
COMMENT ON COLUMN users.auth_provider IS 'Authentication provider: manual, google, or github';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hashed password (NULL for OAuth users)';
COMMENT ON COLUMN users.oauth_provider_id IS 'User ID from OAuth provider';
COMMENT ON COLUMN users.reset_token IS 'Token for password reset (expires after use)';
COMMENT ON COLUMN users.verification_token IS 'Token for email verification';

COMMENT ON TABLE user_sessions IS 'Active user sessions with JWT tokens';
COMMENT ON TABLE login_attempts IS 'Login attempt history for security monitoring';
