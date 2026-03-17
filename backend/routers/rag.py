from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import logging
from pinecone import Pinecone
from clients.bedrock import BedrockClient
import boto3
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

class RagRequest(BaseModel):
    query: str
    session_id: str = "default"
    filters: Optional[Dict[str, Any]] = None

@router.post("/chat")
async def rag_chat(request: RagRequest):
    try:
        query = request.query
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        
        # Generate query embedding
        bedrock = BedrockClient()
        query_vector = bedrock.generate_embedding(query)
        
        # Retrieve chunks
        pc_res = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        
        contexts = []
        sources = []
        seen_res = set()
        for match in pc_res.get('matches', []):
            md = match.get('metadata', {})
            text = md.get('text_content', '')
            if text:
                contexts.append(text)
            
            # Find base id
            raw_chunk_id = match['id']
            base_resource_id = str(md.get('resource_id', raw_chunk_id.split('-chunk')[0]))
            name = md.get('name', '')
            
            # Robust Category Fallback
            category = md.get('category') or md.get('type')
            if not category or str(category).lower() == 'unknown':
                name_lower = name.lower()
                if 'dataset' in name_lower: category = 'dataset'
                elif 'api' in name_lower: category = 'api'
                else: category = 'model'

            if name and base_resource_id not in seen_res:
                seen_res.add(base_resource_id)
                sources.append({
                    "id": raw_chunk_id,
                    "name": name,
                    "category": category,
                    "stars": int(md.get('stars', md.get('github_stars', 0))),
                    "downloads": int(md.get('downloads', md.get('download_count', 0))),
                    "text_content": text,
                    "score": round(match['score'], 4)
                })
        
        context_str = "\n\n".join(contexts) if contexts else "No relevant context found."

        # Generate Answer with synthesis using Amazon Nova Micro (Direct Boto3 to ensure ARN compatibility)
        try:
            
            region = "ap-northeast-1"
            model_id = "arn:aws:bedrock:ap-northeast-1:634319551969:inference-profile/apac.amazon.nova-micro-v1:0"
            
            logger.info(f"🚀 RAG Synthesis: Calling Nova Micro: {model_id}")

            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=region
            )
            
            # Formulate the payload for Nova Micro messages API to match exactly what was verified
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": f"Assistant Instructions: Use the following context to answer the user inquiry. Be professional and use Markdown.\n\nContext: {context_str}\n\nUser Question: {query}"}
                        ]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 1024,
                    "temperature": 0.1
                }
            }
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            response_body = json.loads(response['body'].read())
            # Extract content from Nova response
            answer_text = response_body['output']['message']['content'][0]['text']
            
        except Exception as e:
            logger.error(f"❌ Nova Micro RAG Exception: {type(e).__name__}: {str(e)}")
            # Final Fallback
            answer_text = (
                f"I found the following technical details for you, but the conversational summary engine ({model_id}) encountered an error: {str(e)}\n\n"
                f"**Top Result Detail:**\n\n"
                f"{contexts[0] if contexts else 'No matching details found.'}"
            )
        
        return {
            "answer": answer_text,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"RAG Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
