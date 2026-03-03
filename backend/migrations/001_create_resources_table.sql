-- Migration: Create resources table
-- Description: Main table for storing APIs, Models, and Datasets
-- Requirements: Section 3.1 PostgreSQL Schema

CREATE TABLE IF NOT EXISTS resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'model', 'dataset')),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    long_description TEXT,
    pricing_type VARCHAR(20) NOT NULL CHECK (pricing_type IN ('free', 'paid', 'freemium')),
    price_details JSONB,
    source_url VARCHAR(500) NOT NULL,
    documentation_url VARCHAR(500),
    github_stars INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    health_status VARCHAR(20) DEFAULT 'healthy' CHECK (health_status IN ('healthy', 'degraded', 'down')),
    last_health_check TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT valid_urls CHECK (
        source_url ~ '^https?://' AND
        (documentation_url IS NULL OR documentation_url ~ '^https?://')
    )
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(type);
CREATE INDEX IF NOT EXISTS idx_resources_pricing ON resources(pricing_type);
CREATE INDEX IF NOT EXISTS idx_resources_health ON resources(health_status);
CREATE INDEX IF NOT EXISTS idx_resources_stars ON resources(github_stars DESC);
CREATE INDEX IF NOT EXISTS idx_resources_updated ON resources(updated_at DESC);

-- Add comment for documentation
COMMENT ON TABLE resources IS 'Stores all resources (APIs, Models, Datasets) with metadata and health status';
COMMENT ON COLUMN resources.type IS 'Resource type: api, model, or dataset';
COMMENT ON COLUMN resources.pricing_type IS 'Pricing model: free, paid, or freemium';
COMMENT ON COLUMN resources.health_status IS 'Current health status: healthy, degraded, or down';
COMMENT ON COLUMN resources.metadata IS 'Flexible JSONB field for type-specific data';
