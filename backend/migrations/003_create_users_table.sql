-- Migration: Create users table
-- Description: User profiles with preferences and tech stack
-- Requirements: Section 3.1 PostgreSQL Schema

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cognito_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    preferred_language VARCHAR(10) DEFAULT 'en',
    tech_stack JSONB DEFAULT '[]'::jsonb,
    preferences JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Create indexes for users
CREATE INDEX IF NOT EXISTS idx_users_cognito ON users(cognito_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login DESC);

-- Add comments for documentation
COMMENT ON TABLE users IS 'User profiles with authentication and preferences';
COMMENT ON COLUMN users.cognito_id IS 'AWS Cognito user identifier';
COMMENT ON COLUMN users.preferred_language IS 'User preferred language code (en, hi, etc.)';
COMMENT ON COLUMN users.tech_stack IS 'Array of technologies user works with';
COMMENT ON COLUMN users.preferences IS 'Flexible JSONB field for user preferences';
