#!/usr/bin/env python3
"""
Load backend JSON resource files into the Aurora/PostgreSQL resources table.

This script is intended as a one-off or repeatable backfill utility. It:
- reads local JSON files in the backend directory
- normalizes records into the resources schema
- ensures the resources table supports upsert on (source, source_url)
- upserts records into PostgreSQL/Aurora

Usage:
    python load_json_to_rds.py
    python load_json_to_rds.py --limit 100
    python load_json_to_rds.py --files github_resources.json models.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
from psycopg2.extras import Json, execute_batch

from config import get_database_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = [
    "models.json",
    "huggingface_datasets.json",
    "kaggle_datasets.json",
    "github_resources.json",
]
ALLOWED_TYPES = {"api", "model", "dataset"}
ALLOWED_SOURCES = {"github", "huggingface", "kaggle", "openrouter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load local JSON resources into Aurora/RDS")
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Specific JSON files to ingest from the backend directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of records per file",
    )
    return parser.parse_args()


def load_records(file_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{file_path.name} does not contain a JSON array")
    return data[:limit] if limit else data


def infer_type(record: Dict[str, Any]) -> Optional[str]:
    category = str(record.get("category") or "").lower()
    if category in ALLOWED_TYPES:
        return category

    source = str(record.get("source") or "").lower()
    source_url = str(record.get("source_url") or "").lower()
    tags = [str(tag).lower() for tag in record.get("tags", [])]
    text = " ".join(
        [
            str(record.get("name") or "").lower(),
            str(record.get("description") or "").lower(),
            " ".join(tags),
            source_url,
        ]
    )

    if source == "kaggle" or "/datasets/" in source_url or "dataset" in text:
        return "dataset"
    if any(token in text for token in ["model", "llm", "transformer", "diffusion", "pytorch", "tensorflow"]):
        return "model"
    if any(token in text for token in ["api", "sdk", "rest", "graphql", "client", "fastapi"]):
        return "api"
    return None


def infer_pricing(record: Dict[str, Any]) -> str:
    top_level = str(record.get("pricing_type") or "").lower()
    if top_level in {"free", "paid", "freemium"}:
        return top_level

    metadata_pricing = record.get("metadata", {}).get("pricing_type")
    if isinstance(metadata_pricing, str) and metadata_pricing.lower() in {"free", "paid", "freemium"}:
        return metadata_pricing.lower()

    pricing = record.get("metadata", {}).get("pricing", {})
    if isinstance(pricing, dict):
        prompt = pricing.get("prompt")
        completion = pricing.get("completion")
        try:
            if float(prompt or 0) == 0 and float(completion or 0) == 0:
                return "free"
            return "paid"
        except (TypeError, ValueError):
            return "paid"

    return "free"


def normalize_source(record: Dict[str, Any]) -> str:
    source = str(record.get("source") or "").lower().strip()
    return source if source in ALLOWED_SOURCES else "github"


def normalize_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_url = record.get("source_url")
    name = str(record.get("name") or "").strip()
    description = str(record.get("description") or "").strip()

    if not name or not description or not source_url:
        return None

    resource_type = infer_type(record)
    if resource_type not in ALLOWED_TYPES:
        return None

    source = normalize_source(record)
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "author": record.get("author"),
            "tags": record.get("tags", []),
            "version": record.get("version"),
            "thumbnail_url": record.get("thumbnail_url"),
            "readme_url": record.get("readme_url"),
            "scraped_at": record.get("scraped_at"),
            "original_category": record.get("category"),
        }
    )

    return {
        "type": resource_type,
        "name": name[:255],
        "description": description,
        "long_description": description,
        "pricing_type": infer_pricing(record),
        "price_details": record.get("metadata", {}).get("pricing"),
        "source": source,
        "source_url": source_url,
        "documentation_url": record.get("readme_url") or source_url,
        "github_stars": int(record.get("stars") or 0),
        "download_count": int(record.get("downloads") or 0),
        "active_users": int(record.get("active_users") or 0),
        "health_status": "healthy",
        "language": record.get("language") or metadata.get("language"),
        "private": record.get("private", metadata.get("private")),
        "gated": record.get("gated", metadata.get("gated")),
        "metadata": metadata,
    }


def ensure_upsert_support(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE resources
            ADD COLUMN IF NOT EXISTS source VARCHAR(50) NOT NULL DEFAULT 'github',
            ADD COLUMN IF NOT EXISTS language VARCHAR(100),
            ADD COLUMN IF NOT EXISTS private BOOLEAN,
            ADD COLUMN IF NOT EXISTS gated BOOLEAN
            """
        )
        cur.execute("ALTER TABLE resources DROP CONSTRAINT IF EXISTS chk_resources_source")
        cur.execute(
            """
            ALTER TABLE resources
            ADD CONSTRAINT chk_resources_source
            CHECK (source IN ('github', 'huggingface', 'kaggle', 'openrouter'))
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_source_url_unique
            ON resources(source, source_url)
            """
        )
    conn.commit()


def upsert_records(conn: psycopg2.extensions.connection, records: Iterable[Dict[str, Any]]) -> int:
    payload = [
        (
            record["type"],
            record["name"],
            record["description"],
            record["long_description"],
            record["pricing_type"],
            Json(record["price_details"]) if record["price_details"] is not None else None,
            record["source"],
            record["source_url"],
            record["documentation_url"],
            record["github_stars"],
            record["download_count"],
            record["active_users"],
            record["health_status"],
            record["language"],
            record["private"],
            record["gated"],
            Json(record["metadata"]),
        )
        for record in records
    ]

    if not payload:
        return 0

    with conn.cursor() as cur:
        execute_batch(
            cur,
            """
            INSERT INTO resources (
                type,
                name,
                description,
                long_description,
                pricing_type,
                price_details,
                source,
                source_url,
                documentation_url,
                github_stars,
                download_count,
                active_users,
                health_status,
                language,
                private,
                gated,
                metadata,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (source, source_url)
            DO UPDATE SET
                type = EXCLUDED.type,
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                long_description = EXCLUDED.long_description,
                pricing_type = EXCLUDED.pricing_type,
                price_details = EXCLUDED.price_details,
                documentation_url = EXCLUDED.documentation_url,
                github_stars = EXCLUDED.github_stars,
                download_count = EXCLUDED.download_count,
                active_users = EXCLUDED.active_users,
                health_status = EXCLUDED.health_status,
                language = EXCLUDED.language,
                private = EXCLUDED.private,
                gated = EXCLUDED.gated,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            payload,
            page_size=250,
        )
    conn.commit()
    return len(payload)


def main() -> None:
    args = parse_args()
    database_url = get_database_url()
    if not database_url:
        raise SystemExit("DATABASE_URL is not configured in backend/.env")

    conn = psycopg2.connect(database_url)
    try:
        ensure_upsert_support(conn)
        total_seen = 0
        total_loaded = 0
        total_skipped = 0

        for file_name in args.files:
            file_path = BASE_DIR / file_name
            if not file_path.exists():
                logger.warning("Skipping missing file: %s", file_name)
                continue

            raw_records = load_records(file_path, args.limit)
            normalized_records: List[Dict[str, Any]] = []
            skipped = 0

            for raw_record in raw_records:
                total_seen += 1
                normalized = normalize_record(raw_record)
                if normalized is None:
                    skipped += 1
                    total_skipped += 1
                    continue
                normalized_records.append(normalized)

            loaded = upsert_records(conn, normalized_records)
            total_loaded += loaded
            logger.info(
                "Processed %s: seen=%s loaded=%s skipped=%s",
                file_name,
                len(raw_records),
                loaded,
                skipped,
            )

        logger.info(
            "Done. total_seen=%s total_loaded=%s total_skipped=%s",
            total_seen,
            total_loaded,
            total_skipped,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
