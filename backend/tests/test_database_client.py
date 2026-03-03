"""
Unit tests for database client module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from psycopg2 import OperationalError, DatabaseError

from clients.database import DatabaseClient, DatabaseConnectionError


class TestDatabaseClient:
    """Test suite for DatabaseClient"""
    
    def test_calculate_backoff_delay(self):
        """Test exponential backoff calculation"""
        # Create client without initializing pool by mocking the initialization
        with patch('clients.database.pool.ThreadedConnectionPool'):
            client = DatabaseClient(
                database_url="postgresql://test",
                base_delay=1.0,
                max_delay=30.0
            )
        
        # First attempt: 1 * 2^0 = 1 second
        assert client._calculate_backoff_delay(0) == 1.0
        
        # Second attempt: 1 * 2^1 = 2 seconds
        assert client._calculate_backoff_delay(1) == 2.0
        
        # Third attempt: 1 * 2^2 = 4 seconds
        assert client._calculate_backoff_delay(2) == 4.0
        
        # Large attempt: should be capped at max_delay
        assert client._calculate_backoff_delay(10) == 30.0
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_initialization_success(self, mock_pool):
        """Test successful pool initialization"""
        mock_pool_instance = Mock()
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(
            database_url="postgresql://test",
            pool_size=10,
            max_overflow=5
        )
        
        assert client._pool == mock_pool_instance
        mock_pool.assert_called_once()
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    @patch('clients.database.time.sleep')
    def test_initialization_retry_logic(self, mock_sleep, mock_pool):
        """Test retry logic on initialization failure"""
        # Fail twice, then succeed
        mock_pool.side_effect = [
            OperationalError("Connection failed"),
            OperationalError("Connection failed"),
            Mock()
        ]
        
        client = DatabaseClient(
            database_url="postgresql://test",
            max_retries=3
        )
        
        # Should have retried twice
        assert mock_pool.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_initialization_failure_after_retries(self, mock_pool):
        """Test that initialization fails after max retries"""
        mock_pool.side_effect = OperationalError("Connection failed")
        
        with pytest.raises(DatabaseConnectionError) as exc_info:
            DatabaseClient(
                database_url="postgresql://test",
                max_retries=3
            )
        
        assert "Could not connect to database after 3 attempts" in str(exc_info.value)
        assert mock_pool.call_count == 3
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_get_connection_success(self, mock_pool):
        """Test successful connection retrieval"""
        mock_conn = Mock()
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        with client.get_connection() as conn:
            assert conn == mock_conn
        
        mock_pool_instance.putconn.assert_called_once_with(mock_conn)
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    @patch('clients.database.time.sleep')
    def test_get_connection_retry_logic(self, mock_sleep, mock_pool):
        """Test retry logic when getting connection fails"""
        mock_conn = Mock()
        mock_pool_instance = Mock()
        # Fail once, then succeed
        mock_pool_instance.getconn.side_effect = [
            OperationalError("Connection failed"),
            mock_conn
        ]
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(
            database_url="postgresql://test",
            max_retries=3
        )
        
        with client.get_connection() as conn:
            assert conn == mock_conn
        
        assert mock_pool_instance.getconn.call_count == 2
        assert mock_sleep.call_count == 1
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_execute_query_with_fetch(self, mock_pool):
        """Test query execution with result fetching"""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Test API"},
            {"id": 2, "name": "Test Model"}
        ]
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        results = client.execute_query("SELECT * FROM resources", fetch=True)
        
        assert len(results) == 2
        assert results[0]["name"] == "Test API"
        mock_cursor.execute.assert_called_once()
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_execute_query_without_fetch(self, mock_pool):
        """Test query execution without fetching results"""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        result = client.execute_query(
            "INSERT INTO resources (name) VALUES (%s)",
            params=("Test API",),
            fetch=False
        )
        
        assert result is None
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_execute_query_rollback_on_error(self, mock_pool):
        """Test that transaction is rolled back on error"""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = DatabaseError("Query failed")
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        with pytest.raises(DatabaseError):
            client.execute_query("SELECT * FROM resources")
        
        mock_conn.rollback.assert_called_once()
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_execute_many(self, mock_pool):
        """Test batch query execution"""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        params_list = [
            ("API 1",),
            ("API 2",),
            ("API 3",)
        ]
        
        client.execute_many(
            "INSERT INTO resources (name) VALUES (%s)",
            params_list
        )
        
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_health_check_success(self, mock_pool):
        """Test successful health check"""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        result = client.health_check()
        
        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert result["response_time_ms"] >= 0
        mock_cursor.execute.assert_called_with("SELECT 1")
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_health_check_failure(self, mock_pool):
        """Test health check failure"""
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.side_effect = OperationalError("Connection failed")
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        
        result = client.health_check()
        
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "Could not get database connection" in result["error"]
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_close_pool(self, mock_pool):
        """Test closing the connection pool"""
        mock_pool_instance = Mock()
        mock_pool.return_value = mock_pool_instance
        
        client = DatabaseClient(database_url="postgresql://test")
        client.close()
        
        mock_pool_instance.closeall.assert_called_once()
        assert client._pool is None
    
    @patch('clients.database.pool.ThreadedConnectionPool')
    def test_context_manager(self, mock_pool):
        """Test context manager support"""
        mock_pool_instance = Mock()
        mock_pool.return_value = mock_pool_instance
        
        with DatabaseClient(database_url="postgresql://test") as client:
            assert client._pool == mock_pool_instance
        
        mock_pool_instance.closeall.assert_called_once()
