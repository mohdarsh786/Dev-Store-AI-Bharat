"""
RAG Engine with conversational AI using Bedrock RetrieveAndGenerate
"""
import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

from rag.vector_store import VectorStore
from rag.models import ChatMessage, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine with conversational memory and intent filtering
    """
    
    # Out-of-scope keywords that should be rejected
    OUT_OF_SCOPE_KEYWORDS = [
        'weather', 'news', 'sports', 'politics', 'recipe', 'cooking',
        'health', 'medical', 'legal', 'financial advice', 'stock',
        'personal', 'relationship', 'dating', 'entertainment', 'movie',
        'music recommendation', 'travel', 'hotel', 'restaurant'
    ]
    
    # In-scope keywords for developer tools
    IN_SCOPE_KEYWORDS = [
        'api', 'library', 'framework', 'dataset', 'model', 'tool',
        'sdk', 'package', 'module', 'code', 'programming', 'development',
        'machine learning', 'ai', 'data', 'database', 'github', 'python',
        'javascript', 'java', 'typescript', 'rust', 'go', 'c++', 'ruby'
    ]
    
    def __init__(
        self,
        vector_store: VectorStore,
        bedrock_region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize RAG Engine
        
        Args:
            vector_store: VectorStore instance
            bedrock_region: AWS region
            model_id: Bedrock model ID
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            confidence_threshold: Minimum confidence for answers
        """
        self.vector_store = vector_store
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        
        # Initialize Bedrock clients
        session_kwargs = {'region_name': bedrock_region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            **session_kwargs
        )
        
        self.bedrock_agent_runtime = boto3.client(
            service_name='bedrock-agent-runtime',
            **session_kwargs
        )
        
        logger.info(f"RAGEngine initialized (model={model_id})")
    
    def is_query_in_scope(self, query: str) -> tuple[bool, str]:
        """
        Check if query is about developer tools/resources
        
        Args:
            query: User query
            
        Returns:
            (is_in_scope, reason)
        """
        query_lower = query.lower()
        
        # Check for out-of-scope keywords
        for keyword in self.OUT_OF_SCOPE_KEYWORDS:
            if keyword in query_lower:
                return False, f"Query appears to be about '{keyword}' which is outside my expertise"
        
        # Check for in-scope keywords
        has_in_scope = any(keyword in query_lower for keyword in self.IN_SCOPE_KEYWORDS)
        
        # If query is very short and has no in-scope keywords, it might be off-topic
        if len(query.split()) < 3 and not has_in_scope:
            return False, "Query is too vague or not related to developer tools"
        
        return True, "Query is in scope"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query"""
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('embedding')
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def retrieve_context(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using hybrid search
        
        Args:
            query: Search query
            filters: Optional filters
            max_results: Maximum results
            
        Returns:
            List of relevant documents
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Perform hybrid search
            results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                filters=filters,
                k=max_results,
                vector_weight=0.7,
                bm25_weight=0.3
            )
            
            return results
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def format_context_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for LLM prompt
        
        Args:
            results: Search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant resources found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result['document']
            context_parts.append(
                f"[Resource {i}]\n"
                f"Name: {doc.get('name', 'Unknown')}\n"
                f"Description: {doc.get('description', 'No description')}\n"
                f"Category: {doc.get('category', 'Unknown')}\n"
                f"Source: {doc.get('source', 'Unknown')}\n"
                f"Author: {doc.get('author', 'Unknown')}\n"
                f"Stars: {doc.get('stars', 0)}\n"
                f"Downloads: {doc.get('downloads', 0)}\n"
                f"License: {doc.get('license', 'Unknown')}\n"
                f"URL: {doc.get('source_url', 'N/A')}\n"
                f"Tags: {', '.join(doc.get('tags', [])[:5])}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[ChatMessage]
    ) -> str:
        """
        Build prompt with context and conversation history
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous messages
            
        Returns:
            Complete prompt
        """
        system_prompt = """You are DevStore AI, an expert assistant for developers looking for APIs, models, datasets, and development tools.

Your role:
- Help developers find the right tools, libraries, frameworks, APIs, models, and datasets
- Provide accurate information based ONLY on the retrieved context
- Be concise, technical, and helpful
- Include relevant details like stars, downloads, license, and URLs

Important rules:
1. ONLY answer questions about developer tools, APIs, models, datasets, libraries, and frameworks
2. If the query is not about development tools, politely decline and explain your scope
3. If you don't find relevant information in the context, say "I couldn't find a matching resource in my database"
4. Always cite the source URL when recommending a resource
5. Be honest about limitations - don't make up information
6. Consider conversation history for context

Retrieved Resources:
{context}

Conversation History:
{history}

User Query: {query}

Provide a helpful, accurate response based on the retrieved resources. If no relevant resources were found or the query is out of scope, explain clearly."""
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                history_text += f"{msg.role.upper()}: {msg.content}\n"
        else:
            history_text = "No previous conversation"
        
        return system_prompt.format(
            context=context,
            history=history_text,
            query=query
        )
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using Bedrock
        
        Args:
            prompt: Complete prompt
            max_tokens: Maximum tokens
            temperature: Temperature for generation
            
        Returns:
            Generated response
        """
        try:
            # Prepare request for Claude
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text from Claude response
            content = response_body.get('content', [])
            if content and len(content) > 0:
                return content[0].get('text', '')
            
            return "I apologize, but I couldn't generate a response."
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                logger.warning("Bedrock throttling detected")
                return "I'm experiencing high demand right now. Please try again in a moment."
            elif error_code == 'ValidationException':
                logger.error(f"Validation error: {e}")
                return "I encountered an error processing your request. Please rephrase your question."
            else:
                logger.error(f"Bedrock error: {e}")
                raise
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise
    
    def calculate_confidence(
        self,
        results: List[Dict[str, Any]],
        response: str
    ) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            results: Retrieved results
            response: Generated response
            
        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0
        
        # Base confidence on top result score
        top_score = results[0].get('score', 0) if results else 0
        
        # Normalize score (OpenSearch scores can vary)
        normalized_score = min(top_score / 10.0, 1.0)
        
        # Penalize if response indicates uncertainty
        uncertainty_phrases = [
            "i couldn't find",
            "i don't have",
            "no relevant",
            "not sure",
            "unable to"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            normalized_score *= 0.5
        
        return round(normalized_score, 2)
    
    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Main chat method with RAG
        
        Args:
            request: Chat request
            
        Returns:
            Chat response
        """
        # Check if query is in scope
        is_in_scope, reason = self.is_query_in_scope(request.query)
        
        if not is_in_scope:
            return ChatResponse(
                answer=f"I apologize, but I can only help with questions about developer tools, APIs, models, datasets, and programming resources. {reason}. Please ask me about development tools, libraries, frameworks, or datasets.",
                sources=[],
                confidence=0.0,
                query=request.query
            )
        
        # Retrieve relevant context
        results = self.retrieve_context(
            query=request.query,
            filters=request.filters,
            max_results=request.max_results
        )
        
        # Check if we found relevant results
        if not results:
            return ChatResponse(
                answer="I couldn't find any matching resources in my database for your query. Could you try rephrasing or being more specific about what kind of developer tool, API, model, or dataset you're looking for?",
                sources=[],
                confidence=0.0,
                query=request.query
            )
        
        # Format context
        context = self.format_context_for_prompt(results)
        
        # Build prompt with conversation history
        prompt = self.build_prompt(
            query=request.query,
            context=context,
            conversation_history=request.conversation_history
        )
        
        # Generate response
        answer = self.generate_response(prompt)
        
        # Calculate confidence
        confidence = self.calculate_confidence(results, answer)
        
        # Prepare sources
        sources = [
            {
                "name": r['document'].get('name'),
                "description": r['document'].get('description', '')[:200],
                "url": r['document'].get('source_url'),
                "category": r['document'].get('category'),
                "stars": r['document'].get('stars', 0),
                "score": r.get('score', 0)
            }
            for r in results[:3]  # Top 3 sources
        ]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query=request.query
        )
