# Fix the VectorStore initialization to use the correct index name

old_init = '''    # Initialize vector store
    vector_store = VectorStore(
        opensearch_client=opensearch_client,
        bedrock_client=bedrock_client
    )'''

new_init = '''    # Initialize vector store with correct index name
    from config import settings
    vector_store = VectorStore(
        opensearch_client=opensearch_client,
        bedrock_client=bedrock_client,
        index_name=settings.opensearch_index_name
    )'''

# Read the file
with open('rag/router.py', 'r') as f:
    content = f.read()

# Replace
content = content.replace(old_init, new_init)

# Write back
with open('rag/router.py', 'w') as f:
    f.write(content)

print("Fixed VectorStore initialization in router.py")
