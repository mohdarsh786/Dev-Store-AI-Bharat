-- Migration: Create resource_usage table
-- Description: Track user interactions with resources
-- Requirements: Section 3.1 PostgreSQL Schema

CREATE TABLE IF NOT EXISTS resource_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL CHECK (action IN ('view', 'download_boilerplate', 'test', 'bookmark')),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for resource_usage
CREATE INDEX IF NOT EXISTS idx_usage_user ON resource_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_resource ON resource_usage(resource_id);
CREATE INDEX IF NOT EXISTS idx_usage_action ON resource_usage(action);
CREATE INDEX IF NOT EXISTS idx_usage_created ON resource_usage(created_at DESC);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_usage_user_resource ON resource_usage(user_id, resource_id);

-- Add comments for documentation
COMMENT ON TABLE resource_usage IS 'Track user interactions with resources for analytics and personalization';
COMMENT ON COLUMN resource_usage.action IS 'Type of interaction: view, download_boilerplate, test, or bookmark';
COMMENT ON COLUMN resource_usage.metadata IS 'Additional context about the action';
