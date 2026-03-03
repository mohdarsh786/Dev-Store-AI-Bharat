-- Migration: Create resource_rankings table
-- Description: Store computed ranking scores for resources
-- Requirements: Section 3.1 PostgreSQL Schema

CREATE TABLE IF NOT EXISTS resource_rankings (
    resource_id UUID REFERENCES resources(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    semantic_relevance_avg FLOAT DEFAULT 0,
    popularity_score FLOAT NOT NULL,
    optimization_score FLOAT NOT NULL,
    freshness_score FLOAT NOT NULL,
    final_score FLOAT NOT NULL,
    rank_position INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (resource_id, date)
);

-- Create indexes for resource_rankings
CREATE INDEX IF NOT EXISTS idx_rankings_date ON resource_rankings(date DESC);
CREATE INDEX IF NOT EXISTS idx_rankings_score ON resource_rankings(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_rankings_position ON resource_rankings(rank_position);

-- Composite index for date-based ranking queries
CREATE INDEX IF NOT EXISTS idx_rankings_date_score ON resource_rankings(date DESC, final_score DESC);

-- Add comments for documentation
COMMENT ON TABLE resource_rankings IS 'Daily computed ranking scores for resources';
COMMENT ON COLUMN resource_rankings.semantic_relevance_avg IS 'Average semantic relevance score across queries';
COMMENT ON COLUMN resource_rankings.popularity_score IS 'Normalized popularity score (0-1)';
COMMENT ON COLUMN resource_rankings.optimization_score IS 'Optimization score based on latency, cost, docs (0-1)';
COMMENT ON COLUMN resource_rankings.freshness_score IS 'Freshness score based on recency and health (0-1)';
COMMENT ON COLUMN resource_rankings.final_score IS 'Weighted composite score: 0.4*semantic + 0.3*popularity + 0.2*optimization + 0.1*freshness';
COMMENT ON COLUMN resource_rankings.rank_position IS 'Overall rank position for the date';
