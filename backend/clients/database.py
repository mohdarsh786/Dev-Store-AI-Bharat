"""
Database connection module for DevStore

Provides PostgreSQL connection pool with retry logic and health checks.
"""
import time
import logging
from typing import Optional, Any, Dict, List, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool, OperationalError, DatabaseError
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails after retries"""
    pass


class DatabaseClient:
    """
    PostgreSQL database client with connection pooling and retry logic.
    
    Features:
    - Connection pooling for efficient resource usage
    - Exponential backoff retry logic for transient failures
    - Health check method for monitoring
    - Context manager support for automatic connection cleanup
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = None,
        max_overflow: int = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0
    ):
        """
        Initialize database client with connection pool.
        
        Args:
            database_url: PostgreSQL connection URL (required if not using settings)
            pool_size: Minimum number of connections in pool (defaults to 20)
            max_overflow: Maximum overflow connections (defaults to 10)
            max_retries: Maximum number of retry attempts for failed connections
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds between retries
        """
        # Import settings only when needed to avoid initialization issues in tests
        if database_url is None:
            from config import settings
            database_url = settings.database_url
            pool_size = pool_size or settings.db_pool_size
            max_overflow = max_overflow or settings.db_max_overflow
        
        self.database_url = database_url
        self.pool_size = pool_size or 20
        self.max_overflow = max_overflow or 10
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize the connection pool with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.pool_size + self.max_overflow,
                    dsn=self.database_url
                )
                logger.info(
                    f"Database connection pool initialized successfully "
                    f"(pool_size={self.pool_size}, max_overflow={self.max_overflow})"
                )
                return
            except OperationalError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to initialize database pool after {self.max_retries} attempts: {e}")
                    raise DatabaseConnectionError(
                        f"Could not connect to database after {self.max_retries} attempts"
                    ) from e
                
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds, capped at max_delay
        """
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    @contextmanager
    def get_connection(self):
        """
        Get a database connection from the pool with automatic cleanup.
        
        Usage:
            with db_client.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM resources")
                results = cursor.fetchall()
        
        Yields:
            psycopg2 connection object
            
        Raises:
            DatabaseConnectionError: If connection cannot be obtained after retries
        """
        conn = None
        for attempt in range(self.max_retries):
            try:
                if self._pool is None:
                    raise DatabaseConnectionError("Connection pool not initialized")
                
                conn = self._pool.getconn()
                if conn:
                    yield conn
                    return
            except OperationalError as e:
                if conn:
                    self._pool.putconn(conn, close=True)
                    conn = None
                
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to get connection after {self.max_retries} attempts: {e}")
                    raise DatabaseConnectionError(
                        f"Could not get database connection after {self.max_retries} attempts"
                    ) from e
                
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            finally:
                if conn:
                    self._pool.putconn(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SQL query with automatic connection management.
        
        Args:
            query: SQL query string (use %s for parameters)
            params: Query parameters tuple
            fetch: Whether to fetch and return results
            
        Returns:
            List of result rows as dictionaries if fetch=True, None otherwise
            
        Raises:
            DatabaseConnectionError: If query execution fails after retries
        """
        with self.get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch:
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
                    else:
                        conn.commit()
                        return None
            except DatabaseError as e:
                conn.rollback()
                logger.error(f"Query execution failed: {e}")
                raise
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> None:
        """
        Execute a SQL query multiple times with different parameters.
        
        Useful for batch inserts/updates.
        
        Args:
            query: SQL query string (use %s for parameters)
            params_list: List of parameter tuples
            
        Raises:
            DatabaseConnectionError: If execution fails
        """
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.executemany(query, params_list)
                    conn.commit()
            except DatabaseError as e:
                conn.rollback()
                logger.error(f"Batch execution failed: {e}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Dictionary with health check results:
            {
                "status": "healthy" | "unhealthy",
                "pool_size": int,
                "available_connections": int,
                "response_time_ms": float,
                "error": str (if unhealthy)
            }
        """
        start_time = time.time()
        result = {
            "status": "unhealthy",
            "pool_size": self.pool_size + self.max_overflow,
            "available_connections": 0,
            "response_time_ms": 0.0
        }
        
        try:
            # Try to execute a simple query
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            
            response_time = (time.time() - start_time) * 1000
            result.update({
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            })
            
            logger.info(f"Database health check passed (response_time={response_time:.2f}ms)")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return result
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")
            self._pool = None
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        self.close()


# Global database client instance (can be initialized when needed)
# Usage: from clients.database import DatabaseClient; db_client = DatabaseClient()


# ==================== Async Support ====================

import asyncpg
from typing import Union


class AsyncDatabaseClient:
    """
    Async PostgreSQL database client using asyncpg.
    
    Provides async methods for use with FastAPI and async scripts.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = None,
        max_retries: int = 3
    ):
        """Initialize async database client."""
        if database_url is None:
            from config import settings
            database_url = settings.database_url
            pool_size = pool_size or settings.db_pool_size
        
        self.database_url = database_url
        self.pool_size = pool_size or 20
        self.max_retries = max_retries
        self._pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Establish async connection pool."""
        if self._pool is not None:
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=60
            )
            logger.info(f"Async database pool initialized (size={self.pool_size})")
        except Exception as e:
            logger.error(f"Failed to create async pool: {e}")
            raise DatabaseConnectionError(f"Could not connect to database: {e}") from e
    
    async def disconnect(self) -> None:
        """Close async connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Async database pool closed")
    
    async def execute(
        self,
        query: str,
        *args
    ) -> str:
        """
        Execute a query without returning results.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            *args: Query parameters
            
        Returns:
            Status string from database
        """
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(
        self,
        query: str,
        *args
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            *args: Query parameters
            
        Returns:
            List of rows as dictionaries
        """
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    async def fetchrow(
        self,
        query: str,
        *args
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            *args: Query parameters
            
        Returns:
            Row as dictionary or None
        """
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
    
    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0
    ) -> Any:
        """
        Fetch a single value.
        
        Args:
            query: SQL query with $1, $2, etc. placeholders
            *args: Query parameters
            column: Column index to return
            
        Returns:
            Single value or None
        """
        if not self._pool:
            await self.connect()
        
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column)
    
    async def health_check(self) -> bool:
        """
        Async health check.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._pool:
                await self.connect()
            
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Async health check failed: {e}")
            return False


# Update DatabaseClient to support both sync and async
class DatabaseClient(AsyncDatabaseClient):
    """
    Unified database client supporting both sync and async operations.
    
    Inherits async methods from AsyncDatabaseClient and adds sync methods.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = None,
        max_overflow: int = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0
    ):
        """Initialize unified database client."""
        # Initialize async parent
        super().__init__(database_url, pool_size, max_retries)
        
        # Initialize sync pool
        self.max_overflow = max_overflow or 10
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sync_pool: Optional[pool.ThreadedConnectionPool] = None
    
    def _initialize_sync_pool(self) -> None:
        """Initialize synchronous connection pool."""
        if self._sync_pool is not None:
            return
        
        for attempt in range(self.max_retries):
            try:
                self._sync_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.pool_size + self.max_overflow,
                    dsn=self.database_url
                )
                logger.info(
                    f"Sync database pool initialized "
                    f"(pool_size={self.pool_size}, max_overflow={self.max_overflow})"
                )
                return
            except OperationalError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to initialize sync pool after {self.max_retries} attempts: {e}")
                    raise DatabaseConnectionError(
                        f"Could not connect to database after {self.max_retries} attempts"
                    ) from e
                
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    @contextmanager
    def get_connection(self):
        """Get a sync database connection from the pool."""
        if self._sync_pool is None:
            self._initialize_sync_pool()
        
        conn = None
        for attempt in range(self.max_retries):
            try:
                conn = self._sync_pool.getconn()
                if conn:
                    yield conn
                    return
            except OperationalError as e:
                if conn:
                    self._sync_pool.putconn(conn, close=True)
                    conn = None
                
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to get connection after {self.max_retries} attempts: {e}")
                    raise DatabaseConnectionError(
                        f"Could not get database connection after {self.max_retries} attempts"
                    ) from e
                
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
            finally:
                if conn:
                    self._sync_pool.putconn(conn)
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute a SQL query synchronously."""
        with self.get_connection() as conn:
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if fetch:
                        results = cursor.fetchall()
                        return [dict(row) for row in results]
                    else:
                        conn.commit()
                        return None
            except DatabaseError as e:
                conn.rollback()
                logger.error(f"Query execution failed: {e}")
                raise
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> None:
        """Execute a SQL query multiple times synchronously."""
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.executemany(query, params_list)
                    conn.commit()
            except DatabaseError as e:
                conn.rollback()
                logger.error(f"Batch execution failed: {e}")
                raise
    
    def close(self) -> None:
        """Close sync connection pool."""
        if self._sync_pool:
            self._sync_pool.closeall()
            logger.info("Sync database pool closed")
            self._sync_pool = None
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context manager."""
        self.close()
