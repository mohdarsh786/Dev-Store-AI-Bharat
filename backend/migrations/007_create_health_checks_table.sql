-- Migration: Create health_checks table
-- Description: Store health check results for resources
-- Requirements: Section 3.1 PostgreSQL Schema

CREATE TABLE IF NOT EXISTS health_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'down')),
    response_time_ms INTEGER,
    error_message TEXT,
    checked_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for health_checks
CREATE INDEX IF NOT EXISTS idx_health_resource ON health_checks(resource_id);
CREATE INDEX IF NOT EXISTS idx_health_checked ON health_checks(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_health_status ON health_checks(status);

-- Composite index for resource health history queries
CREATE INDEX IF NOT EXISTS idx_health_resource_checked ON health_checks(resource_id, checked_at DESC);

-- Add comments for documentation
COMMENT ON TABLE health_checks IS 'Historical health check results for resources';
COMMENT ON COLUMN health_checks.status IS 'Health status: healthy, degraded, or down';
COMMENT ON COLUMN health_checks.response_time_ms IS 'Response time in milliseconds';
COMMENT ON COLUMN health_checks.error_message IS 'Error message if health check failed';
COMMENT ON COLUMN health_checks.checked_at IS 'Timestamp when health check was performed';
