# Read the file
with open('clients/opensearch.py', 'r') as f:
    lines = f.readlines()

# Find the health_check function and replace it
new_method = '''    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the OpenSearch connection.
        
        Returns:
            Dictionary with health check results:
            {
                "status": "healthy" | "unhealthy",
                "cluster_name": str,
                "cluster_status": str,
                "response_time_ms": float,
                "error": str (if unhealthy)
            }
        """
        start_time = time.time()
        result = {
            "status": "unhealthy",
            "response_time_ms": 0.0
        }
        
        try:
            # For OpenSearch Serverless, check if we can list indices
            is_serverless = "aoss.amazonaws.com" in self.host
            
            if is_serverless:
                # Serverless doesn't support cluster health endpoint
                # Just check if we can list indices
                try:
                    indices = self._client.cat.indices(format='json')
                    response_time = (time.time() - start_time) * 1000
                    result.update({
                        "status": "healthy",
                        "cluster_name": "OpenSearch Serverless",
                        "cluster_status": "available",
                        "indices_count": len(indices),
                        "response_time_ms": round(response_time, 2)
                    })
                    logger.info(
                        f"OpenSearch Serverless health check passed "
                        f"(indices={len(indices)}, response_time={response_time:.2f}ms)"
                    )
                except NotFoundError:
                    # Even cat.indices might not work, try a simple index exists check
                    exists = self._client.indices.exists(index=self.index_name)
                    response_time = (time.time() - start_time) * 1000
                    result.update({
                        "status": "healthy",
                        "cluster_name": "OpenSearch Serverless",
                        "cluster_status": "available",
                        "index_exists": exists,
                        "response_time_ms": round(response_time, 2)
                    })
                    logger.info(
                        f"OpenSearch Serverless health check passed "
                        f"(index_exists={exists}, response_time={response_time:.2f}ms)"
                    )
            else:
                # Standard OpenSearch cluster
                health = self._client.cluster.health()
                info = self._client.info()
                
                response_time = (time.time() - start_time) * 1000
                result.update({
                    "status": "healthy",
                    "cluster_name": info.get('cluster_name'),
                    "cluster_status": health.get('status'),
                    "response_time_ms": round(response_time, 2)
                })
                
                logger.info(
                    f"OpenSearch health check passed "
                    f"(cluster_status={health.get('status')}, response_time={response_time:.2f}ms)"
                )
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"OpenSearch health check failed: {e}")
        
        return result
    
'''

# Find where health_check starts
start_idx = None
for i, line in enumerate(lines):
    if 'def health_check(self)' in line:
        start_idx = i
        break

if start_idx is None:
    print("Could not find health_check function")
    exit(1)

# Find where the next method starts (close method)
end_idx = None
for i in range(start_idx + 1, len(lines)):
    if lines[i].strip().startswith('def close(self)'):
        end_idx = i
        break

if end_idx is None:
    print("Could not find end of health_check function")
    exit(1)

# Replace the method
new_lines = lines[:start_idx] + [new_method] + lines[end_idx:]

# Write back
with open('clients/opensearch.py', 'w') as f:
    f.writelines(new_lines)

print(f"Fixed health_check method (lines {start_idx+1} to {end_idx})")
