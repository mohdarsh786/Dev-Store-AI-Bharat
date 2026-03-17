"""
Document Chunking Service for RAG

Uses langchain-text-splitters to break large documents into readable chunks
for vector retrieval.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:
    """
    Splits resource descriptions and names into smaller, manageable chunks
    for optimized vector store retrieval.
    """

    def __init__(self):
        """Initialize semantic chunker with appropriate parameters"""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_resource(self, resource_data: dict) -> list[dict]:
        """
        Split a single resource into multiple document chunks while 
        retaining critical metadata.
        
        Args:
            resource_data: Dictionary containing resource details
            
        Returns:
            List of chunked dictionary blocks with parent metadata
        """
        name = resource_data.get("name", "")
        description = resource_data.get("description", "")
        
        content = f"{name}\n\n{description}"
        if not content.strip():
            return []

        # Split content using LangChain
        text_chunks = self.splitter.split_text(content)
        
        chunked_resources = []
        for index, chunk_text in enumerate(text_chunks):
            # Retain original Pinecone metadata and append the chunk text
            chunk_doc = {
                "id": f"{resource_data.get('id', 'unknown')}-chunk-{index}",
                "resource_id": resource_data.get("id"),
                "name": name,
                "category": resource_data.get("category", ""),
                "stars": resource_data.get("stars", 0),
                "downloads": resource_data.get("downloads", 0),
                "tags": resource_data.get("tags", []),
                "text_content": chunk_text,
                "chunk_index": index
            }
            chunked_resources.append(chunk_doc)
            
        return chunked_resources
