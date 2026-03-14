# Remove doc_id from index_document call

old_index_call = '''                    # Index in OpenSearch
                    self.opensearch.index_document(
                        document=document,
                        doc_id=str(resource_dict['id']),
                        refresh=False
                    )'''

new_index_call = '''                    # Index in OpenSearch (Serverless doesn't support custom doc_id)
                    self.opensearch.index_document(
                        document=document,
                        refresh=False
                    )'''

# Read the file
with open('rag/ingestor.py', 'r') as f:
    content = f.read()

# Replace the index call
content = content.replace(old_index_call, new_index_call)

# Write back
with open('rag/ingestor.py', 'w') as f:
    f.write(content)

print("Fixed doc_id issue in ingestor.py")
