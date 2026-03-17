import logging
import json
import os
import boto3
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class BedrockClient:
    def __init__(self):
        self.bedrock_region = os.getenv("BEDROCK_REGION", "ap-northeast-1")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
        self.embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")

        try:
            from botocore.config import Config
            config = Config(
                connect_timeout=3,
                read_timeout=3,
                retries={'max_attempts': 0} # No retries for search-time latency
            )
            self._client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.bedrock_region,
                config=config
            )
            logger.info(f"Nova Engine Ready | Region: {self.bedrock_region} | Model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock: {e}")

    def invoke_claude(self, messages: List[Dict[str, str]], system: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Using AWS Converse API - Compatible with both Claude and Nova"""
        try:
            formatted_messages = []
            for m in messages:
                if m['role'] not in ['system']:
                    formatted_messages.append({
                        "role": m['role'],
                        "content": [{"text": m['content']}]
                    })

            sys_prompt = system or os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
            
            response = self._client.converse(
                modelId=self.model_id,
                messages=formatted_messages,
                system=[{"text": sys_prompt}],
                inferenceConfig={
                    "temperature": kwargs.get('temperature', 0.1),
                    "maxTokens": kwargs.get('max_tokens', 1000)
                }
            )
            
            return {
                "content": [{"text": response['output']['message']['content'][0]['text']}]
            }
        except Exception as e:
            logger.error(f"Bedrock Error: {e}")
            return {"content": [{"text": "Maaf kijiye, main connect nahi kar pa raha hoon."}]}

    def generate_embedding(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text})
        response = self._client.invoke_model(modelId=self.embedding_model_id, body=body)
        return json.loads(response['body'].read())['embedding']
