# Add database ingestion method to the DataIngestor class

db_method = '''
    def ingest_from_database(self) -> Dict[str, Any]:
        """
        Ingest resources directly from PostgreSQL database.
        
        Returns:
            Statistics dict with counts
        """
        import psycopg2
        from config import settings
        
        stats = {
            'total_records': 0,
            'total_indexed': 0,
            'total_failed': 0
        }
        
        try:
            # Connect to database
            logger.info("Connecting to PostgreSQL database...")
            conn = psycopg2.connect(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password
            )
            cursor = conn.cursor()
            
            # Fetch all resources
            logger.info("Fetching resources from database...")
            cursor.execute("""
                SELECT id, name, description, type, url, documentation_url, 
                       pricing_type, github_stars, downloads, last_updated,
                       health_status, source
                FROM resources
                ORDER BY id
            """)
            
            rows = cursor.fetchall()
            stats['total_records'] = len(rows)
            logger.info(f"Found {len(rows)} resources in database")
            
            # Process in batches
            batch = []
            for row in rows:
                try:
                    # Map database row to resource dict
                    resource_dict = {
                        'id': row[0],
                        'name': row[1] or 'Unknown',
                        'description': row[2] or 'No description available',
                        'category': row[3].lower() if row[3] else 'api',  # type -> category
                        'source': row[11] or 'github',
                        'source_url': row[4] or '',
                        'author': 'Unknown',
                        'stars': row[7] or 0,
                        'downloads': row[8] or 0,
                        'license': None,
                        'tags': [],
                        'metadata': {
                            'documentation_url': row[5],
                            'pricing_type': row[6],
                            'health_status': row[10],
                            'last_updated': row[9].isoformat() if row[9] else None
                        }
                    }
                    
                    # Create searchable text
                    text = f"Name: {resource_dict['name']} | Description: {resource_dict['description']} | Category: {resource_dict['category']} | Source: {resource_dict['source']}"
                    
                    # Generate embedding
                    embedding = self.generate_embedding(text)
                    
                    # Prepare document for OpenSearch
                    document = {
                        'name': resource_dict['name'],
                        'description': resource_dict['description'][:1000],
                        'resource_type': resource_dict['category'].upper(),
                        'pricing_type': resource_dict['metadata']['pricing_type'] or 'free',
                        'source': resource_dict['source'],
                        'source_url': resource_dict['source_url'],
                        'author': resource_dict['author'],
                        'github_stars': resource_dict['stars'],
                        'downloads': resource_dict['downloads'],
                        'license': resource_dict['license'] or 'Unknown',
                        'tags': resource_dict['tags'],
                        'last_updated': resource_dict['metadata']['last_updated'] or datetime.utcnow().isoformat(),
                        'health_status': resource_dict['metadata']['health_status'] or 'healthy',
                        'embedding': embedding,
                        'metadata': resource_dict['metadata']
                    }
                    
                    # Index in OpenSearch
                    self.opensearch.index_document(
                        document=document,
                        doc_id=str(resource_dict['id']),
                        refresh=False
                    )
                    
                    stats['total_indexed'] += 1
                    
                    if stats['total_indexed'] % 10 == 0:
                        logger.info(f"Indexed {stats['total_indexed']}/{stats['total_records']} resources...")
                    
                except Exception as e:
                    logger.error(f"Failed to process resource {row[0]}: {e}")
                    stats['total_failed'] += 1
                    continue
            
            cursor.close()
            conn.close()
            
            # Refresh index to make documents searchable
            logger.info("Refreshing OpenSearch index...")
            self.opensearch._client.indices.refresh(index=self.opensearch.index_name)
            
            logger.info(f"Database ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Database ingestion failed: {e}")
            raise
'''

# Read the file
with open('rag/ingestor.py', 'r') as f:
    content = f.read()

# Find where to insert (before the last line or before ingest_all method)
insert_pos = content.rfind('    def ingest_all(self,')

if insert_pos == -1:
    print("Could not find insertion point")
    exit(1)

# Insert the new method
new_content = content[:insert_pos] + db_method + '\n' + content[insert_pos:]

# Write back
with open('rag/ingestor.py', 'w') as f:
    f.write(new_content)

print("Added ingest_from_database method to DataIngestor class")
