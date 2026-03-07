-- Migration: Add ingestion pipeline support schema
-- Description: Adds source identity, embedding state, run tracking, and scheduler support columns
-- Date: 2026-03-07

ALTER TABLE resources ADD COLUMN IF NOT EXISTS source VARCHAR(50) NOT NULL DEFAULT 'manual';
ALTER TABLE resources ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
ALTER TABLE resources ADD COLUMN IF NOT EXISTS embedding JSONB;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS embedding_hash VARCHAR(64);
ALTER TABLE resources ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMP;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS source_updated_at TIMESTAMP;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS last_ingested_at TIMESTAMP;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS last_indexed_at TIMESTAMP;
ALTER TABLE resources ADD COLUMN IF NOT EXISTS rank_score FLOAT DEFAULT 0.0;

CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_source_url_unique ON resources(source, source_url);
CREATE INDEX IF NOT EXISTS idx_resources_source ON resources(source);
CREATE INDEX IF NOT EXISTS idx_resources_content_hash ON resources(content_hash);
CREATE INDEX IF NOT EXISTS idx_resources_embedding_hash ON resources(embedding_hash);
CREATE INDEX IF NOT EXISTS idx_resources_rank_score ON resources(rank_score DESC);

COMMENT ON COLUMN resources.source IS 'External source for resource ingestion: github, huggingface, kaggle, openrouter, etc.';
COMMENT ON COLUMN resources.tags IS 'Normalized tags used for search and embedding generation';
COMMENT ON COLUMN resources.content_hash IS 'Canonical content hash used for ingestion idempotency';
COMMENT ON COLUMN resources.embedding IS 'Serialized embedding vector stored as JSONB for index rebuilds';
COMMENT ON COLUMN resources.embedding_hash IS 'Hash of the embedding input text';
COMMENT ON COLUMN resources.embedding_updated_at IS 'Timestamp of the last embedding generation';
COMMENT ON COLUMN resources.source_updated_at IS 'Timestamp reported by the upstream source, when available';
COMMENT ON COLUMN resources.last_ingested_at IS 'Timestamp when the resource was last processed by the ingestion orchestrator';
COMMENT ON COLUMN resources.last_indexed_at IS 'Timestamp when the resource document was last upserted into OpenSearch';
COMMENT ON COLUMN resources.rank_score IS 'Current global ranking score for browse and trending surfaces';

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    source VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('running', 'success', 'partial_success', 'failed', 'skipped')),
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMP,
    stage VARCHAR(50),
    fetched_count INTEGER NOT NULL DEFAULT 0,
    inserted_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    unchanged_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    embedded_count INTEGER NOT NULL DEFAULT 0,
    indexed_count INTEGER NOT NULL DEFAULT 0,
    top_failure_reason TEXT,
    partial_completion BOOLEAN NOT NULL DEFAULT FALSE,
    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_run_id ON ingestion_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source ON ingestion_runs(source);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at ON ingestion_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_status ON ingestion_runs(status);

COMMENT ON TABLE ingestion_runs IS 'Operational tracking records for orchestrated ingestion runs, including source-level progress and counters';
COMMENT ON COLUMN ingestion_runs.run_id IS 'Shared run identifier across the top-level orchestration record and its source-level children';
COMMENT ON COLUMN ingestion_runs.progress IS 'Structured counters and stage metadata for operational visibility';
