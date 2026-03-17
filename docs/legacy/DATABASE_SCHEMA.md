# Dev-Store Database Schema

Complete schema for all tables in the Dev-Store AI/ML Resource Discovery Platform.

---

-- TABLE 1: resources
CREATE TABLE resources (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    category        VARCHAR(50) NOT NULL,
    source_url      VARCHAR(500),
    license         VARCHAR(100),
    price_type      VARCHAR(20) DEFAULT 'free',
    stars           INTEGER DEFAULT 0,
    downloads       INTEGER DEFAULT 0,
    activity_score  DOUBLE PRECISION DEFAULT 0.0,
    author          VARCHAR(255),
    tags            TEXT,
    version         VARCHAR(50),
    thumbnail_url   VARCHAR(500),
    readme_url      VARCHAR(500),
    rank_score      DOUBLE PRECISION DEFAULT 0.0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP
);

-- TABLE 2: categories
CREATE TABLE categories (
    id              SERIAL PRIMARY KEY,
    slug            VARCHAR(100) UNIQUE NOT NULL,
    label           VARCHAR(100) NOT NULL,
    description     TEXT,
    resource_count  INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 3: tags
CREATE TABLE tags (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) UNIQUE NOT NULL,
    slug            VARCHAR(100) UNIQUE NOT NULL,
    usage_count     INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 4: resource_tags
CREATE TABLE resource_tags (
    resource_id     INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    tag_id          INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (resource_id, tag_id)
);

-- TABLE 5: embeddings
CREATE TABLE embeddings (
    id              SERIAL PRIMARY KEY,
    resource_id     INTEGER UNIQUE NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    vector          TEXT NOT NULL,
    model_used      VARCHAR(100) DEFAULT 'placeholder',
    dimensions      INTEGER DEFAULT 384,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP
);

-- TABLE 6: bundles
CREATE TABLE bundles (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    use_case        VARCHAR(255),
    graph_json      TEXT,
    is_featured     BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 7: bundle_resources
CREATE TABLE bundle_resources (
    bundle_id       INTEGER NOT NULL REFERENCES bundles(id) ON DELETE CASCADE,
    resource_id     INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    role            VARCHAR(50),
    order_index     INTEGER DEFAULT 0,
    PRIMARY KEY (bundle_id, resource_id)
);

-- TABLE 8: boilerplate_requests
CREATE TABLE boilerplate_requests (
    id              SERIAL PRIMARY KEY,
    resource_id     INTEGER REFERENCES resources(id) ON DELETE SET NULL,
    bundle_id       INTEGER REFERENCES bundles(id) ON DELETE SET NULL,
    language        VARCHAR(50) DEFAULT 'python',
    framework       VARCHAR(50),
    generated_code  TEXT,
    ip_hash         VARCHAR(64),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 9: search_logs
CREATE TABLE search_logs (
    id              SERIAL PRIMARY KEY,
    query           VARCHAR(500) NOT NULL,
    category        VARCHAR(50),
    price_type      VARCHAR(20),
    results_count   INTEGER DEFAULT 0,
    clicked_id      INTEGER REFERENCES resources(id) ON DELETE SET NULL,
    session_id      VARCHAR(100),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 10: ingestion_logs
CREATE TABLE ingestion_logs (
    id               SERIAL PRIMARY KEY,
    source           VARCHAR(100) NOT NULL,
    status           VARCHAR(50) DEFAULT 'pending',
    records_fetched  INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_skipped  INTEGER DEFAULT 0,
    error_message    TEXT,
    started_at       TIMESTAMP,
    finished_at      TIMESTAMP,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TABLE 11: rankings_cache
CREATE TABLE rankings_cache (
    id              SERIAL PRIMARY KEY,
    cache_key       VARCHAR(255) UNIQUE NOT NULL,
    data_json       TEXT NOT NULL,
    expires_at      TIMESTAMP NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

---

## Entity Relationship Overview

```
resources ──────────── resource_tags ──── tags
    │
    ├───────────────── embeddings
    │
    ├───────────────── bundle_resources ── bundles
    │
    ├───────────────── boilerplate_requests
    │
    └───────────────── search_logs (clicked_id)

ingestion_logs  (standalone)
rankings_cache  (standalone)
categories      (standalone)
```

---

## Implementation Priority

| Priority | Table               | Phase   | Reason                                  |
|----------|---------------------|---------|-----------------------------------------|
| 1        | `resources`         | Now     | Already exists — add 6 new columns      |
| 2        | `tags` + `resource_tags` | Phase 4 | Used by search and filter          |
| 3        | `bundles` + `bundle_resources` | Phase 4 | Blueprint page needs it      |
| 4        | `embeddings`        | Phase 5 | RAG / semantic search                   |
| 5        | `boilerplate_requests` | Phase 6 | Boilerplate generator               |
| 6        | `ingestion_logs`    | Phase 4 | Ingestion pipeline tracking             |
| 7        | `search_logs`       | Phase 7 | Analytics and usage tracking            |
| 8        | `rankings_cache`    | Phase 3 | Performance — cache ranked results      |
| 9        | `categories`        | Any     | Nice to have — dynamic category config  |
