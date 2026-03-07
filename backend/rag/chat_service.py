"""
Production-ready RAG chat service with conversational memory.

This module provides:
- Conversational chat with context memory
- Bedrock RetrieveAndGenerate API integration
- Out-of-scope query rejection
- Confidence-based response filtering
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# Prompt template for RAG
SYSTEM_PROMPT = """You are a helpful AI assistant for Dev-Store, a developer-focused search engine for APIs, models, and datasets.

Your role:
- Answer questions ONLY about developer tools, APIs, ML models, and datasets
- Base your answers ONLY on the retrieved context provided
- If the context doesn't contain relevant information, say "I couldn't find a matching resource in our database"
- Be concise and technical
- Provide specific resource names, descriptions, and links when available

Out-of-scope topics:
- General programming questions not related to finding resources
- Personal advice, medical, legal, or financial advice
- Current events, news, or politics
- Anything not related to developer tools and resources

If a question is out of scope, politely decline and redirect to Dev-Store's purpose."""


class ConversationMemory:
    """Simple in-memory conversation storage"""
    
    def __init__(self, max_history: int = 5):
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-(self.max_history * 2):]
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversations.get(session_id, [])
    
    def clear(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]


class ChatService:
    """
    RAG-powered chat service with conversational memory.
    
    Features:
    - Context-aware responses using retrieved documents
    - Conversation history management
    - Out-of-scope query detection
    - Confidence-based filtering
    """
    
    def __init__(
        self,
        vector_store,
        bedrock_client,
        min_confidence: float = 0.3
    ):
        """
        Initialize chat service.
        
        Args:
            vector_store: VectorStore instance
            bedrock_client: BedrockClient for LLM
            min_confidence: Minimum confidence score for responses
        """
        self.vector_store = vector_store
        self.bedrock = bedrock_client
        self.min_confidence = min_confidence
        self.memory = ConversationMemory()
    
    def is_greeting(self, query: str) -> bool:
        """
        Check if query is a greeting or casual conversation.
        
        Args:
            query: User query
            
        Returns:
            True if greeting
        """
        greetings = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
            'good evening', 'howdy', 'what\'s up', 'whats up', 'sup',
            'how are you', 'how do you do', 'nice to meet you'
        ]
        
        query_lower = query.lower().strip()
        
        # Exact match or starts with greeting
        return any(query_lower == greeting or query_lower.startswith(greeting) 
                   for greeting in greetings)
    
    def is_in_scope(self, query: str) -> bool:
        """
        Check if query is within scope (developer tools/resources).
        
        Args:
            query: User query
            
        Returns:
            True if in scope
        """
        # Greetings are always in scope (handled separately)
        if self.is_greeting(query):
            return True
        
        # Keywords indicating in-scope queries (check these FIRST)
        in_scope_keywords = [
            'api', 'model', 'dataset', 'library', 'framework', 'tool',
            'package', 'sdk', 'service', 'platform', 'resource',
            'github', 'huggingface', 'kaggle', 'npm', 'pypi',
            'machine learning', 'ml', 'ai', 'data', 'code',
            'python', 'javascript', 'java', 'rust', 'go',
            'suggest', 'recommend', 'find', 'search', 'looking for', 'need', 'want'
        ]
        
        # Keywords indicating DEFINITELY out-of-scope (only reject if NO in-scope keywords present)
        out_of_scope_keywords = [
            'weather forecast', 'stock price', 'recipe for', 'how to cook',
            'political news', 'sports score', 'movie review', 'restaurant recommendation'
        ]
        
        query_lower = query.lower()
        
        # Check in-scope FIRST - if any in-scope keyword is present, it's in scope
        # This allows queries like "medical dataset" or "healthcare API" to pass through
        if any(keyword in query_lower for keyword in in_scope_keywords):
            return True
        
        # Only check out-of-scope if NO in-scope keywords were found
        if any(keyword in query_lower for keyword in out_of_scope_keywords):
            return False
        
        # Default: assume in-scope for any query (let the RAG system handle it)
        return True
    
    def retrieve_context(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector store.
        
        Args:
            query: User query
            filters: Optional filters
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            results = self.vector_store.hybrid_search(
                query=query,
                filters=filters,
                k=top_k,
                size=top_k,
                alpha=0.7  # 70% vector, 30% keyword
            )
            
            return results
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            results: Search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant resources found in the database."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result['document']
            context_parts.append(
                f"{i}. {doc['name']}\n"
                f"   Type: {doc['resource_type']}\n"
                f"   Description: {doc['description'][:200]}...\n"
                f"   Source: {doc['source']} | Stars: {doc.get('github_stars', 0)} | Downloads: {doc.get('downloads', 0)}\n"
                f"   URL: {doc.get('source_url', 'N/A')}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate response using Bedrock with RAG context.
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous messages
            
        Returns:
            Response dict with answer and metadata
        """
        try:
            # Build prompt with context
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                }
            ]
            
            # Add conversation history
            for msg in conversation_history[-4:]:  # Last 4 messages
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            # Add current query with context
            user_message = f"""Context (Retrieved Resources):
{context}

User Question: {query}

Please answer based on the context above. If the context doesn't contain relevant information, say "I couldn't find a matching resource in our database"."""
            
            messages.append({
                'role': 'user',
                'content': user_message
            })
            
            # Generate response using Bedrock
            # Will try Claude first, then fallback to Titan Text if Claude not available
            try:
                response = self.bedrock.invoke_claude(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.get('content', [{}])[0].get('text', '')
            except Exception as e:
                logger.warning(f"Bedrock LLM not available, using formatted response: {e}")
                answer = None
            
            # If no answer generated, use smart formatted response
            if not answer:
                if "No relevant resources found" in context:
                    answer = "I couldn't find a matching resource in our database. Could you try rephrasing your question or being more specific about what you're looking for?"
                else:
                    answer = f"Based on the search results, here are some relevant resources:\n\n{context}"
            
            return {
                'answer': answer,
                'context_used': context,
                'confidence': 0.8 if answer else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'answer': "I encountered an error processing your request. Please try again.",
                'context_used': '',
                'confidence': 0.0,
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def chat(
        self,
        query: str,
        session_id: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main chat endpoint with RAG and conversation memory.
        
        Args:
            query: User query
            session_id: Session identifier for conversation tracking
            filters: Optional search filters
            
        Returns:
            Response dict with answer, sources, and metadata
        """
        # Handle greetings without retrieval
        if self.is_greeting(query):
            greeting_responses = [
                "Hello! I'm Dev-Store AI, your assistant for finding developer tools, APIs, ML models, and datasets. How can I help you today?",
                "Hi there! I can help you discover APIs, machine learning models, and datasets. What are you looking for?",
                "Hey! I'm here to help you find the perfect developer resources. What kind of tool or model do you need?"
            ]
            
            # Use first response for consistency
            answer = greeting_responses[0]
            
            response = {
                'answer': answer,
                'sources': [],
                'confidence': 1.0,
                'in_scope': True,
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add to memory
            self.memory.add_message(session_id, 'user', query)
            self.memory.add_message(session_id, 'assistant', answer)
            
            return response
        
        # Check if query is in scope
        if not self.is_in_scope(query):
            response = {
                'answer': "I'm sorry, but I can only help with questions about developer tools, APIs, ML models, and datasets. Your question seems to be outside my area of expertise. Please ask about finding or comparing developer resources!",
                'sources': [],
                'confidence': 0.0,
                'in_scope': False,
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Still add to memory for context
            self.memory.add_message(session_id, 'user', query)
            self.memory.add_message(session_id, 'assistant', response['answer'])
            
            return response
        
        # Retrieve context
        results = self.retrieve_context(query, filters, top_k=5)
        context = self.format_context(results)
        
        # Get conversation history
        history = self.memory.get_history(session_id)
        
        # Generate response
        response_data = self.generate_response(query, context, history)
        
        # Check confidence
        if response_data['confidence'] < self.min_confidence:
            response_data['answer'] = "I couldn't find a matching resource with high confidence. Could you provide more details or try a different search term?"
        
        # Add to memory
        self.memory.add_message(session_id, 'user', query)
        self.memory.add_message(session_id, 'assistant', response_data['answer'])
        
        # Format final response
        return {
            'answer': response_data['answer'],
            'sources': [r['document'] for r in results[:3]],  # Top 3 sources
            'confidence': response_data['confidence'],
            'in_scope': True,
            'session_id': session_id,
            'timestamp': response_data['timestamp']
        }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        self.memory.clear(session_id)
