"""Aurora/PostgreSQL repository methods for ingestion persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from psycopg2.extras import Json, RealDictCursor

from clients.database import DatabaseClient
from models import CanonicalResource, IngestionCounters, IngestionStatus


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "uncategorized"


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


@dataclass
class UpsertResult:
    resource_id: str
    action: str
    row: Dict[str, Any]
    embedding_changed: bool


class IngestionRepository:
    """Persistence layer for canonical resource ingestion and run tracking."""

    def __init__(self, db: DatabaseClient | None = None):
        self.db = db or DatabaseClient()

    def create_run_record(self, run_id: UUID, source: str, status: IngestionStatus) -> Dict[str, Any]:
        now = datetime.utcnow()
        query = """
            INSERT INTO ingestion_runs (run_id, source, status, started_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
        """
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (str(run_id), source, status.value, now, now))
                row = dict(cursor.fetchone())
                conn.commit()
                return row

    def update_run_record(
        self,
        run_id: UUID,
        source: str,
        *,
        status: Optional[IngestionStatus] = None,
        stage: Optional[str] = None,
        counters: Optional[IngestionCounters] = None,
        top_failure_reason: Optional[str] = None,
        partial_completion: Optional[bool] = None,
        progress: Optional[Dict[str, Any]] = None,
        finished: bool = False,
    ) -> Dict[str, Any]:
        now = datetime.utcnow()
        query = """
            UPDATE ingestion_runs
            SET
                status = COALESCE(%s, status),
                stage = COALESCE(%s, stage),
                fetched_count = COALESCE(%s, fetched_count),
                inserted_count = COALESCE(%s, inserted_count),
                updated_count = COALESCE(%s, updated_count),
                unchanged_count = COALESCE(%s, unchanged_count),
                failed_count = COALESCE(%s, failed_count),
                embedded_count = COALESCE(%s, embedded_count),
                indexed_count = COALESCE(%s, indexed_count),
                top_failure_reason = COALESCE(%s, top_failure_reason),
                partial_completion = COALESCE(%s, partial_completion),
                progress = COALESCE(%s, progress),
                finished_at = CASE WHEN %s THEN %s ELSE finished_at END,
                updated_at = %s
            WHERE run_id = %s AND source = %s
            RETURNING *
        """
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    query,
                    (
                        status.value if status else None,
                        stage,
                        counters.fetched_count if counters else None,
                        counters.inserted_count if counters else None,
                        counters.updated_count if counters else None,
                        counters.unchanged_count if counters else None,
                        counters.failed_count if counters else None,
                        counters.embedded_count if counters else None,
                        counters.indexed_count if counters else None,
                        top_failure_reason,
                        partial_completion,
                        Json(progress) if progress is not None else None,
                        finished,
                        now,
                        now,
                        str(run_id),
                        source,
                    ),
                )
                row = dict(cursor.fetchone())
                conn.commit()
                return row

    def latest_run_status(self, source: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = """
            SELECT *
            FROM ingestion_runs
            WHERE (%s IS NULL OR source = %s)
            ORDER BY started_at DESC
            LIMIT 1
        """
        rows = self.db.execute_query(query, (source, source))
        return rows[0] if rows else None

    def upsert_resource(self, resource: CanonicalResource) -> UpsertResult:
        existing_rows = self.db.execute_query(
            """
            SELECT *
            FROM resources
            WHERE source = %s AND source_url = %s
            LIMIT 1
            """,
            (_enum_value(resource.source), resource.source_url),
        )
        now = datetime.utcnow()
        content_hash = resource.content_hash()
        embedding_hash = resource.embedding_hash()

        if existing_rows:
            existing = existing_rows[0]
            if existing.get("content_hash") == content_hash:
                with self.db.get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute(
                            """
                            UPDATE resources
                            SET last_ingested_at = %s
                            WHERE id = %s
                            RETURNING *
                            """,
                            (now, existing["id"]),
                        )
                        row = dict(cursor.fetchone())
                        conn.commit()
                        return UpsertResult(str(row["id"]), "unchanged", row, False)

            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        UPDATE resources
                        SET
                            type = %s,
                            name = %s,
                            description = %s,
                            long_description = %s,
                            pricing_type = %s,
                            source = %s,
                            documentation_url = %s,
                            github_stars = %s,
                            download_count = %s,
                            active_users = %s,
                            health_status = %s,
                            updated_at = %s,
                            metadata = %s,
                            tags = %s,
                            content_hash = %s,
                            embedding_hash = %s,
                            source_updated_at = %s,
                            last_ingested_at = %s
                        WHERE id = %s
                        RETURNING *
                        """,
                        (
                            _enum_value(resource.resource_type),
                            resource.name,
                            resource.description,
                            resource.description,
                            _enum_value(resource.pricing_type),
                            _enum_value(resource.source),
                            resource.documentation_url,
                            resource.github_stars,
                            resource.download_count,
                            resource.active_users,
                            _enum_value(resource.health_status),
                            now,
                            Json(resource.metadata),
                            Json(resource.tags),
                            content_hash,
                            embedding_hash,
                            resource.source_updated_at,
                            now,
                            existing["id"],
                        ),
                    )
                    row = dict(cursor.fetchone())
                    conn.commit()
            self._sync_categories(row["id"], resource.categories, _enum_value(resource.resource_type))
            return UpsertResult(
                str(row["id"]),
                "updated",
                row,
                existing.get("embedding_hash") != embedding_hash or not existing.get("embedding"),
            )

        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO resources (
                        type, name, description, long_description, pricing_type, price_details,
                        source, source_url, documentation_url, github_stars, download_count,
                        active_users, health_status, created_at, updated_at, metadata, tags,
                        content_hash, embedding_hash, source_updated_at, last_ingested_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        _enum_value(resource.resource_type),
                        resource.name,
                        resource.description,
                        resource.description,
                        _enum_value(resource.pricing_type),
                        Json(resource.metadata.get("model", {}).get("pricing", {})),
                        _enum_value(resource.source),
                        resource.source_url,
                        resource.documentation_url,
                        resource.github_stars,
                        resource.download_count,
                        resource.active_users,
                        _enum_value(resource.health_status),
                        now,
                        now,
                        Json(resource.metadata),
                        Json(resource.tags),
                        content_hash,
                        embedding_hash,
                        resource.source_updated_at,
                        now,
                    ),
                )
                row = dict(cursor.fetchone())
                conn.commit()
        self._sync_categories(row["id"], resource.categories, _enum_value(resource.resource_type))
        return UpsertResult(str(row["id"]), "inserted", row, True)

    def _sync_categories(self, resource_id: UUID | str, categories: List[str], resource_type: str) -> None:
        if not categories:
            return
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                category_ids: List[str] = []
                for category in categories:
                    slug = _slugify(category)
                    cursor.execute(
                        """
                        INSERT INTO categories (name, slug, resource_type, display_order)
                        VALUES (%s, %s, %s, 0)
                        ON CONFLICT (slug) DO UPDATE
                        SET name = EXCLUDED.name, resource_type = EXCLUDED.resource_type
                        RETURNING id
                        """,
                        (category, slug, resource_type),
                    )
                    category_ids.append(str(cursor.fetchone()["id"]))

                cursor.execute("DELETE FROM resource_categories WHERE resource_id = %s", (str(resource_id),))
                for category_id in category_ids:
                    cursor.execute(
                        """
                        INSERT INTO resource_categories (resource_id, category_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (str(resource_id), category_id),
                    )
                conn.commit()

    def update_embedding(self, resource_id: str, embedding: List[float], embedding_hash: str) -> Dict[str, Any]:
        now = datetime.utcnow()
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    UPDATE resources
                    SET embedding = %s, embedding_hash = %s, embedding_updated_at = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (Json(embedding), embedding_hash, now, now, resource_id),
                )
                row = dict(cursor.fetchone())
                conn.commit()
                return row

    def mark_indexed(self, resource_id: str) -> None:
        now = datetime.utcnow()
        self.db.execute_query(
            """
            UPDATE resources
            SET last_indexed_at = %s, updated_at = %s
            WHERE id = %s
            """,
            (now, now, resource_id),
            fetch=False,
        )

    def fetch_resources_for_ranking(self) -> List[Dict[str, Any]]:
        return self.db.execute_query(
            """
            SELECT
                id,
                type,
                source,
                name,
                description,
                source_url,
                documentation_url,
                pricing_type,
                github_stars,
                download_count,
                active_users,
                health_status,
                updated_at,
                source_updated_at,
                metadata
            FROM resources
            """
        ) or []

    def persist_rankings(self, rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranking_date = date.today()
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                for item in rankings:
                    cursor.execute(
                        """
                        UPDATE resources
                        SET rank_score = %s, trending_score = %s, category_rank = %s, updated_at = %s
                        WHERE id = %s
                        """,
                        (
                            item["rank_score"],
                            item["trending_score"],
                            item["category_rank"],
                            datetime.utcnow(),
                            item["id"],
                        ),
                    )
                    cursor.execute(
                        """
                        INSERT INTO resource_rankings (
                            resource_id, date, semantic_relevance_avg, popularity_score,
                            optimization_score, freshness_score, final_score, rank_position,
                            created_at, trending_score, category_rank
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (resource_id, date) DO UPDATE
                        SET
                            semantic_relevance_avg = EXCLUDED.semantic_relevance_avg,
                            popularity_score = EXCLUDED.popularity_score,
                            optimization_score = EXCLUDED.optimization_score,
                            freshness_score = EXCLUDED.freshness_score,
                            final_score = EXCLUDED.final_score,
                            rank_position = EXCLUDED.rank_position,
                            trending_score = EXCLUDED.trending_score,
                            category_rank = EXCLUDED.category_rank
                        """,
                        (
                            item["id"],
                            ranking_date,
                            0.5,
                            item["popularity_score"],
                            item["optimization_score"],
                            item["freshness_score"],
                            item["final_score"],
                            item["rank_position"],
                            datetime.utcnow(),
                            item["trending_score"],
                            item["category_rank"],
                        ),
                    )
                conn.commit()
        return {
            "ranked_count": len(rankings),
            "top_ranked_id": rankings[0]["id"] if rankings else None,
            "ranking_date": ranking_date.isoformat(),
        }

    def list_trending_resources(
        self, 
        resource_type: Optional[str] = None, 
        pricing_type: Optional[str] = None,
        sort_by: Optional[str] = None,
        limit: int = 40
    ) -> List[Dict[str, Any]]:
        """Get trending resources with comprehensive filtering support."""
        
        # Build WHERE clause conditions
        where_conditions = []
        params = []
        
        if resource_type:
            where_conditions.append("type = %s")
            params.append(resource_type)
            
        if pricing_type:
            where_conditions.append("pricing_type = %s")
            params.append(pricing_type)
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Build ORDER BY clause
        if sort_by == "downloads":
            order_clause = "ORDER BY download_count DESC, github_stars DESC"
        elif sort_by == "popularity":
            order_clause = "ORDER BY github_stars DESC, download_count DESC"
        elif sort_by == "paid":
            order_clause = "ORDER BY github_stars DESC, rank_score DESC"
        else:
            order_clause = "ORDER BY rank_score DESC, trending_score DESC, github_stars DESC"
        
        query = f"""
            SELECT
                id,
                name,
                description,
                type AS resource_type,
                pricing_type,
                github_stars,
                download_count AS downloads,
                documentation_url,
                health_status,
                updated_at AS last_updated,
                rank_score,
                category_rank,
                trending_score
            FROM resources
            {where_clause}
            {order_clause}
            LIMIT %s
        """
        
        params.append(limit)
        return self.db.execute_query(query, tuple(params)) or []

    def list_all_resources(self, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """List all resources from database"""
        query = """
            SELECT 
                id,
                type as category,
                name,
                description,
                source,
                source_url,
                documentation_url,
                pricing_type,
                github_stars as stars,
                download_count as downloads,
                active_users,
                health_status,
                tags,
                metadata,
                rank_score,
                trending_score,
                category_rank,
                created_at,
                updated_at
            FROM resources
            ORDER BY rank_score DESC NULLS LAST, created_at DESC
            LIMIT %s OFFSET %s
        """
        return self.db.execute_query(query, (limit, offset)) or []
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource statistics from database"""
        query = """
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE type IN ('model', 'api')) as models,
                COUNT(*) FILTER (WHERE type = 'dataset') as datasets,
                COUNT(*) FILTER (WHERE type = 'repository') as repositories,
                COUNT(*) FILTER (WHERE source = 'huggingface') as huggingface,
                COUNT(*) FILTER (WHERE source = 'openrouter') as openrouter,
                COUNT(*) FILTER (WHERE source = 'github') as github,
                COUNT(*) FILTER (WHERE source = 'kaggle') as kaggle
            FROM resources
        """
        result = self.db.execute_query(query)
        if result and len(result) > 0:
            row = result[0]
            return {
                "total_resources": row.get("total", 0),
                "models": row.get("models", 0),
                "datasets": row.get("datasets", 0),
                "repositories": row.get("repositories", 0),
                "by_source": {
                    "huggingface": row.get("huggingface", 0),
                    "openrouter": row.get("openrouter", 0),
                    "github": row.get("github", 0),
                    "kaggle": row.get("kaggle", 0)
                },
                "source": "database"
            }
        return {
            "total_resources": 0,
            "models": 0,
            "datasets": 0,
            "repositories": 0,
            "by_source": {},
            "source": "database"
        }
