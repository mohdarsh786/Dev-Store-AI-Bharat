-- Migration: Create search_history table
-- Description: Track user search queries and interactions
-- Requirements: Section 3.1 PostgreSQL Schema

CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    language VARCHAR(10),
    filters JSONB,
    results_count INTEGER,
    clicked_resources UUID[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for search_history
CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(user_id);
CREATE INDEX IF NOT EXISTS idx_search_created ON search_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_language ON search_history(language);

-- Add comments for documentation
COMMENT ON TABLE search_history IS 'User search history for personalization and analytics';
COMMENT ON COLUMN search_history.query IS 'Natural language search query';
COMMENT ON COLUMN search_history.language IS 'Language of the query (en, hi, etc.)';
COMMENT ON COLUMN search_history.filters IS 'Applied filters as JSONB';
COMMENT ON COLUMN search_history.clicked_resources IS 'Array of resource IDs user clicked on';
