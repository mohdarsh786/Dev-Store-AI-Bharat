"""
Unit tests for OpenSearch client module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from opensearchpy.exceptions import (
    ConnectionError as OpenSearchConnectionError,
    TransportError,
    NotFoundError
)

from clients.opensearch import OpenSearchClient, OpenSearchClientError


class TestOpenSearchClient:
    """Test suite for OpenSearchClient"""
    
    def test_calculate_backoff_delay(self):
        """Test exponential backoff calculation"""
        # Create client without initializing by mocking the initialization
        with patch('clients.opensearch.OpenSearch'):
            client = OpenSearchClient(
                host="localhost",
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
    
    @patch('clients.opensearch.OpenSearch')
    def test_initialization_success(self, mock_opensearch):
        """Test successful client initialization"""
        mock_client = Mock()
        mock_client.info.return_value = {
            'cluster_name': 'test-cluster',
            'version': {'number': '2.4.0'}
        }
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(
            host="localhost",
            port=9200,
            use_ssl=False
        )
        
        assert client._client == mock_client
        mock_client.info.assert_called_once()
    
    @patch('clients.opensearch.OpenSearch')
    @patch('clients.opensearch.time.sleep')
    def test_initialization_retry_logic(self, mock_sleep, mock_opensearch):
        """Test retry logic on initialization failure"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test-cluster'}
        
        # Fail twice, then succeed
        mock_opensearch.side_effect = [
            OpenSearchConnectionError("N/A", "Connection failed", Exception("Connection failed")),
            OpenSearchConnectionError("N/A", "Connection failed", Exception("Connection failed")),
            mock_client
        ]
        
        client = OpenSearchClient(
            host="localhost",
            max_retries=3
        )
        
        # Should have retried twice
        assert mock_opensearch.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('clients.opensearch.OpenSearch')
    def test_initialization_failure_after_retries(self, mock_opensearch):
        """Test that initialization fails after max retries"""
        mock_opensearch.side_effect = OpenSearchConnectionError("N/A", "Connection failed", Exception("Connection failed"))
        
        with pytest.raises(OpenSearchClientError) as exc_info:
            OpenSearchClient(
                host="localhost",
                max_retries=3
            )
        
        assert "Could not connect to OpenSearch after 3 attempts" in str(exc_info.value)
        assert mock_opensearch.call_count == 3
    
    @patch('clients.opensearch.OpenSearch')
    def test_index_exists_true(self, mock_opensearch):
        """Test checking if index exists (returns True)"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = True
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        assert client.index_exists("test_index") is True
        mock_client.indices.exists.assert_called_once_with(index="test_index")
    
    @patch('clients.opensearch.OpenSearch')
    def test_index_exists_false(self, mock_opensearch):
        """Test checking if index exists (returns False)"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = False
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        assert client.index_exists("nonexistent_index") is False
    
    @patch('clients.opensearch.OpenSearch')
    def test_create_index_success(self, mock_opensearch):
        """Test successful index creation"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {'acknowledged': True}
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        mapping = {
            'properties': {
                'name': {'type': 'text'},
                'embedding': {'type': 'knn_vector', 'dimension': 1536}
            }
        }
        
        result = client.create_index("test_index", mapping=mapping)
        
        assert result is True
        mock_client.indices.create.assert_called_once()
    
    @patch('clients.opensearch.OpenSearch')
    def test_create_index_already_exists(self, mock_opensearch):
        """Test creating index that already exists"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = True
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        result = client.create_index("existing_index")
        
        assert result is False
        mock_client.indices.create.assert_not_called()
    
    @patch('clients.opensearch.OpenSearch')
    def test_create_index_failure(self, mock_opensearch):
        """Test index creation failure"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = TransportError("N/A", "Creation failed", Exception("Creation failed"))
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        with pytest.raises(OpenSearchClientError) as exc_info:
            client.create_index("test_index")
        
        assert "Could not create index" in str(exc_info.value)
    
    @patch('clients.opensearch.OpenSearch')
    def test_delete_index_success(self, mock_opensearch):
        """Test successful index deletion"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = True
        mock_client.indices.delete.return_value = {'acknowledged': True}
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        result = client.delete_index("test_index")
        
        assert result is True
        mock_client.indices.delete.assert_called_once_with(index="test_index")
    
    @patch('clients.opensearch.OpenSearch')
    def test_delete_index_not_exists(self, mock_opensearch):
        """Test deleting index that doesn't exist"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.indices.exists.return_value = False
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        result = client.delete_index("nonexistent_index")
        
        assert result is False
        mock_client.indices.delete.assert_not_called()
    
    @patch('clients.opensearch.OpenSearch')
    def test_index_document_success(self, mock_opensearch):
        """Test successful document indexing"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.index.return_value = {
            '_id': 'doc123',
            'result': 'created'
        }
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        document = {
            'name': 'Test API',
            'description': 'A test API',
            'embedding': [0.1] * 1536
        }
        
        result = client.index_document(document, doc_id="doc123")
        
        assert result['_id'] == 'doc123'
        mock_client.index.assert_called_once()
    
    @patch('clients.opensearch.OpenSearch')
    def test_index_document_failure(self, mock_opensearch):
        """Test document indexing failure"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.index.side_effect = TransportError("N/A", "Indexing failed", Exception("Indexing failed"))
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        document = {'name': 'Test API'}
        
        with pytest.raises(OpenSearchClientError) as exc_info:
            client.index_document(document)
        
        assert "Could not index document" in str(exc_info.value)
    
    @patch('clients.opensearch.OpenSearch')
    def test_knn_search_success(self, mock_opensearch):
        """Test successful KNN vector search"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.search.return_value = {
            'hits': {
                'hits': [
                    {
                        '_id': 'doc1',
                        '_score': 0.95,
                        '_source': {
                            'name': 'API 1',
                            'description': 'First API'
                        }
                    },
                    {
                        '_id': 'doc2',
                        '_score': 0.85,
                        '_source': {
                            'name': 'API 2',
                            'description': 'Second API'
                        }
                    }
                ]
            }
        }
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        query_vector = [0.1] * 1536
        results = client.knn_search(query_vector, k=10)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc1'
        assert results[0]['score'] == 0.95
        assert results[0]['document']['name'] == 'API 1'
        mock_client.search.assert_called_once()
    
    @patch('clients.opensearch.OpenSearch')
    def test_knn_search_with_filters(self, mock_opensearch):
        """Test KNN search with filters"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.search.return_value = {'hits': {'hits': []}}
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        query_vector = [0.1] * 1536
        filters = {
            'pricing_type': ['free', 'freemium'],
            'health_status': 'healthy'
        }
        
        results = client.knn_search(query_vector, k=10, filters=filters)
        
        # Verify the search was called with filters
        call_args = mock_client.search.call_args
        query_body = call_args[1]['body']
        
        assert 'filter' in query_body['query']['bool']
        assert len(results) == 0
    
    @patch('clients.opensearch.OpenSearch')
    def test_knn_search_failure(self, mock_opensearch):
        """Test KNN search failure"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.search.side_effect = TransportError("N/A", "Search failed", Exception("Search failed"))
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        query_vector = [0.1] * 1536
        
        with pytest.raises(OpenSearchClientError) as exc_info:
            client.knn_search(query_vector)
        
        assert "Could not perform KNN search" in str(exc_info.value)
    
    @patch('clients.opensearch.OpenSearch')
    def test_health_check_success(self, mock_opensearch):
        """Test successful health check"""
        mock_client = Mock()
        mock_client.info.return_value = {
            'cluster_name': 'test-cluster',
            'version': {'number': '2.4.0'}
        }
        mock_client.cluster.health.return_value = {
            'status': 'green',
            'number_of_nodes': 3
        }
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        result = client.health_check()
        
        assert result['status'] == 'healthy'
        assert result['cluster_name'] == 'test-cluster'
        assert result['cluster_status'] == 'green'
        assert 'response_time_ms' in result
        assert result['response_time_ms'] >= 0
    
    @patch('clients.opensearch.OpenSearch')
    def test_health_check_failure(self, mock_opensearch):
        """Test health check failure"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.cluster.health.side_effect = OpenSearchConnectionError("N/A", "Connection failed", Exception("Connection failed"))
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        result = client.health_check()
        
        assert result['status'] == 'unhealthy'
        assert 'error' in result
    
    @patch('clients.opensearch.OpenSearch')
    def test_close_client(self, mock_opensearch):
        """Test closing the client"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        client.close()
        
        assert client._client is None
    
    @patch('clients.opensearch.OpenSearch')
    def test_context_manager(self, mock_opensearch):
        """Test context manager support"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_opensearch.return_value = mock_client
        
        with OpenSearchClient(host="localhost") as client:
            assert client._client == mock_client
        
        assert client._client is None
    
    @patch('clients.opensearch.OpenSearch')
    def test_default_index_name(self, mock_opensearch):
        """Test that default index name is used"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost", index_name="custom_index")
        
        assert client.index_name == "custom_index"
    
    @patch('clients.opensearch.OpenSearch')
    def test_knn_search_empty_results(self, mock_opensearch):
        """Test KNN search with no results"""
        mock_client = Mock()
        mock_client.info.return_value = {'cluster_name': 'test'}
        mock_client.search.return_value = {'hits': {'hits': []}}
        mock_opensearch.return_value = mock_client
        
        client = OpenSearchClient(host="localhost")
        
        query_vector = [0.1] * 1536
        results = client.knn_search(query_vector, k=10)
        
        assert len(results) == 0
        assert isinstance(results, list)
