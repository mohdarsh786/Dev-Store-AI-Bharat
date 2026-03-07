"""
Bedrock client module for DevStore

Provides AWS Bedrock connection for text generation and embeddings.
"""
import time
import logging
import json
from typing import Optional, Dict, Any, List
import boto3
from botocore.exceptions import ClientError, BotoCoreError

logger = logging.getLogger(__name__)


class BedrockClientError(Exception):
    """Raised when Bedrock operations fail"""
    pass


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e


class BedrockClient:
    """
    AWS Bedrock client for text generation and embeddings.
    
    Features:
    - Text generation using Claude 3
    - Embedding generation using Titan Embeddings
    - Retry logic with exponential backoff
    - Circuit breaker pattern for fault tolerance
    """
    
    def __init__(
        self,
        region_name: Optional[str] = None,
        model_id: Optional[str] = None,
        embedding_model_id: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region_name: AWS region (defaults to us-east-1)
            model_id: Model ID for text generation
            embedding_model_id: Model ID for embeddings
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout: Seconds before attempting recovery
        """
        if region_name is None:
            from config import settings
            region_name = settings.aws_region
            model_id = model_id or settings.bedrock_model_id
            embedding_model_id = embedding_model_id or settings.bedrock_embedding_model_id
        
        self.region_name = region_name
        self.model_id = model_id or "anthropic.claude-3-sonnet-20240229-v1:0"
        self.embedding_model_id = embedding_model_id or "amazon.titan-embed-text-v1"
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        self._client = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout=circuit_breaker_timeout
        )
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Bedrock client."""
        try:
            # Get credentials from settings if available
            from config import settings
            
            # Configure boto3 session with credentials
            import boto3
            session_kwargs = {'region_name': self.region_name}
            
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                session_kwargs['aws_access_key_id'] = settings.aws_access_key_id
                session_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key
                logger.info("Using AWS credentials from settings")
            else:
                logger.info("Using default AWS credentials (IAM role or environment)")
            
            session = boto3.Session(**session_kwargs)
            self._client = session.client('bedrock-runtime')
            
            logger.info(f"Bedrock client initialized (region={self.region_name})")
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise BedrockClientError("Could not initialize Bedrock client") from e
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using Claude 3.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
            
        Raises:
            BedrockClientError: If generation fails after retries
            CircuitBreakerOpen: If circuit breaker is open
        """
        def _invoke():
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "top_p": top_p
            })
            
            for attempt in range(self.max_retries):
                try:
                    response = self._client.invoke_model(
                        modelId=self.model_id,
                        body=body
                    )
                    
                    response_body = json.loads(response['body'].read())
                    text = response_body['content'][0]['text']
                    logger.debug(f"Text generated successfully ({len(text)} chars)")
                    return text
                    
                except (ClientError, BotoCoreError) as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Text generation failed after {self.max_retries} attempts: {e}")
                        raise BedrockClientError("Could not generate text") from e
                    
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        return self._circuit_breaker.call(_invoke)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector using Titan Embeddings.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector (list of floats)
            
        Raises:
            BedrockClientError: If embedding generation fails
            CircuitBreakerOpen: If circuit breaker is open
        """
        def _invoke():
            body = json.dumps({
                "inputText": text
            })
            
            for attempt in range(self.max_retries):
                try:
                    response = self._client.invoke_model(
                        modelId=self.embedding_model_id,
                        body=body
                    )
                    
                    response_body = json.loads(response['body'].read())
                    embedding = response_body['embedding']
                    logger.debug(f"Embedding generated successfully ({len(embedding)} dimensions)")
                    return embedding
                    
                except (ClientError, BotoCoreError) as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Embedding generation failed after {self.max_retries} attempts: {e}")
                        raise BedrockClientError("Could not generate embedding") from e
                    
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        return self._circuit_breaker.call(_invoke)
    
    def invoke_claude(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke Claude with messages (for RAG chat).
        Falls back to Titan Text if Claude is not available.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: Optional system prompt
            
        Returns:
            Response dict from Claude or Titan
        """
        def _invoke():
            # Try Claude first
            try:
                return self._invoke_claude_internal(messages, max_tokens, temperature, system)
            except Exception as e:
                logger.warning(f"Claude not available, falling back to Titan Text: {e}")
                # Fallback to Titan Text
                return self._invoke_titan_text(messages, max_tokens, temperature, system)
        
        return self._circuit_breaker.call(_invoke)
    
    def _invoke_claude_internal(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        system: Optional[str]
    ) -> Dict[str, Any]:
        """Internal method to invoke Claude."""
        # Filter out system messages and use as system prompt
        filtered_messages = []
        system_prompt = system
        
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                filtered_messages.append(msg)
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": filtered_messages,
            "temperature": temperature
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        logger.debug(f"Claude invoked successfully")
        return response_body
    
    def _invoke_titan_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        system: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback method to invoke Amazon Titan Text."""
        # Combine messages into a single prompt for Titan
        prompt_parts = []
        
        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.append(f"System: {msg['content']}\n")
            elif msg['role'] == 'user':
                prompt_parts.append(f"User: {msg['content']}\n")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"Assistant: {msg['content']}\n")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
                "stopSequences": []
            }
        }
        
        # Try Titan Text Lite first (available in ap-northeast-3)
        titan_models = [
            "amazon.titan-text-lite-v1",
            "amazon.titan-text-express-v1"
        ]
        
        last_error = None
        for model_id in titan_models:
            try:
                response = self._client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response['body'].read())
                
                # Format response to match Claude's structure
                text = response_body['results'][0]['outputText'].strip()
                logger.info(f"Titan Text ({model_id}) invoked successfully")
                
                return {
                    "content": [{"text": text}],
                    "model": model_id
                }
            except Exception as e:
                logger.warning(f"Failed to invoke {model_id}: {e}")
                last_error = e
                continue
        
        # If all models failed, raise the last error
        raise BedrockClientError(f"All Titan Text models failed: {last_error}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Bedrock connection.
        
        Returns:
            Health check results dictionary
        """
        start_time = time.time()
        result = {
            "status": "unhealthy",
            "circuit_breaker_state": self._circuit_breaker.state,
            "response_time_ms": 0.0
        }
        
        try:
            # Test with a simple embedding generation
            self.generate_embedding("health check")
            
            response_time = (time.time() - start_time) * 1000
            result.update({
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            })
            logger.info(f"Bedrock health check passed (response_time={response_time:.2f}ms)")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Bedrock health check failed: {e}")
        
        return result
