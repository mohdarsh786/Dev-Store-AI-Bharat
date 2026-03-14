"""
Ingestion Run Tracker

Tracks ingestion run status and statistics
"""
from datetime import datetime
from typing import Dict, Any, Optional


class RunTracker:
    """
    Tracks ingestion run status in database
    
    Stores:
    - Run metadata (ID, timestamps, status)
    - Statistics (fetched, inserted, updated, failed counts)
    - Per-source statistics
    - Failure reasons
    """
    
    def __init__(self, db_client):
        """
        Initialize run tracker
        
        Args:
            db_client: Database client
        """
        self.db = db_client
    
    def create_run(self, run_id: str, sources, status: str = "running") -> str:
        """
        Create new ingestion run record
        
        Args:
            run_id: Unique run identifier
            sources: List of enabled sources or a single source name
            status: Initial run status
            
        Returns:
            Run ID
        """
        query = """
            INSERT INTO ingestion_runs (
                run_id,
                sources,
                status,
                started_at
            ) VALUES (%s, %s, %s, NOW())
            RETURNING run_id
        """
        
        import json
        if isinstance(sources, str):
            sources = [sources]

        result = self.db.execute(
            query,
            (run_id, json.dumps(sources), status)
        )
        
        return result['run_id']
    
    def update_run_status(
        self,
        run_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """
        Update run status
        
        Args:
            run_id: Run identifier
            status: New status (running, completed, failed)
            error: Error message if failed
        """
        query = """
            UPDATE ingestion_runs
            SET
                status = %s,
                error_message = %s,
                finished_at = CASE
                    WHEN %s IN ('completed', 'failed') THEN NOW()
                    ELSE finished_at
                END
            WHERE run_id = %s
        """
        
        self.db.execute(query, (status, error, status, run_id))
    
    def update_run_stats(
        self,
        run_id: str,
        stats: Dict[str, int],
        source_stats: Dict[str, Dict[str, int]] = None
    ):
        """
        Update run statistics
        
        Args:
            run_id: Run identifier
            stats: Overall statistics
            source_stats: Per-source statistics
        """
        import json
        
        query = """
            UPDATE ingestion_runs
            SET
                fetched_count = %s,
                inserted_count = %s,
                updated_count = %s,
                failed_count = %s,
                skipped_count = %s,
                source_stats = %s
            WHERE run_id = %s
        """
        
        self.db.execute(query, (
            stats.get('fetched_count', 0),
            stats.get('inserted_count', 0),
            stats.get('updated_count', 0),
            stats.get('failed_count', 0),
            stats.get('skipped_count', 0),
            json.dumps(source_stats or {}),
            run_id
        ))
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run details"""
        query = """
            SELECT *
            FROM ingestion_runs
            WHERE run_id = %s
        """
        
        return self.db.fetch_one(query, (run_id,))
    
    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Get latest ingestion run"""
        query = """
            SELECT *
            FROM ingestion_runs
            ORDER BY started_at DESC
            LIMIT 1
        """
        
        return self.db.fetch_one(query)
    
    def get_latest_successful_run(self) -> Optional[Dict[str, Any]]:
        """Get latest successful ingestion run"""
        query = """
            SELECT *
            FROM ingestion_runs
            WHERE status = 'completed'
            ORDER BY finished_at DESC
            LIMIT 1
        """
        
        return self.db.fetch_one(query)
    
    def get_run_history(
        self,
        limit: int = 10,
        status: Optional[str] = None
    ) -> list:
        """
        Get ingestion run history
        
        Args:
            limit: Maximum number of runs to return
            status: Filter by status
            
        Returns:
            List of run records
        """
        if status:
            query = """
                SELECT *
                FROM ingestion_runs
                WHERE status = %s
                ORDER BY started_at DESC
                LIMIT %s
            """
            return self.db.fetch_all(query, (status, limit))
        else:
            query = """
                SELECT *
                FROM ingestion_runs
                ORDER BY started_at DESC
                LIMIT %s
            """
            return self.db.fetch_all(query, (limit,))

    def update_run(
        self,
        run_id: str,
        source=None,
        status: Optional[str] = None,
        counters: Optional[Dict[str, int]] = None,
        top_failure_reason: Optional[str] = None,
        finished: bool = False,
    ):
        """
        Backward-compatible wrapper used by the pipeline/tests.
        """
        if counters:
            normalized_stats = {
                "fetched_count": counters.get("fetched_count", 0),
                "inserted_count": counters.get("inserted_count", 0),
                "updated_count": counters.get("updated_count", 0),
                "failed_count": counters.get("failed_count", 0),
                "skipped_count": counters.get("skipped_count", 0),
            }
            self.update_run_stats(run_id, normalized_stats)

        if status is not None:
            normalized_status = status.value if hasattr(status, "value") else status
            if finished and normalized_status == "success":
                normalized_status = "completed"
            elif finished and normalized_status == "failed":
                normalized_status = "failed"

            self.update_run_status(run_id, normalized_status, top_failure_reason)
