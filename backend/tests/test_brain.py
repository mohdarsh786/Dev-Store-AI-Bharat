import boto3
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

# Load from environment variables
AWS_ACCESS_KEY = settings.aws_access_key_id
AWS_SECRET_KEY = settings.aws_secret_access_key
CLAUDE_ARN = os.getenv('BEDROCK_CLAUDE_ARN', settings.bedrock_model_id)
REGION = settings.aws_region 

def test_osaka_bedrock():
    print(f"🚀 Initializing Bedrock in {REGION}...")
    
    try:
        # Initialize the client with your keys and the Osaka region
        bedrock = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=REGION
        )

        # TEST 1: Titan Embeddings (Ensure they are enabled in ap-northeast-3)
        embedding_body = json.dumps({
            "inputText": "Testing DevStore search intent",
            "dimensions": 1024,
            "normalize": True
        })
        
        bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=embedding_body
        )
        print("✅ Success: Titan Embeddings are live in Osaka!")

        # TEST 2: Claude 3.5 Sonnet (Using the Converse API as per your CLI)
        messages = [{"role": "user", "content": [{"text": "Bhai, ek bar 'System Online' bol do!"}]}]
        
        response = bedrock.converse(
            modelId=CLAUDE_ARN,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 1
            }
        )
        
        result_text = response['output']['message']['content'][0]['text']
        print(f"✅ Success: Claude says: '{result_text}'")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_osaka_bedrock()