"""
Lazy exports for AWS and external service clients.
"""

__all__ = [
    "DatabaseClient",
    "DatabaseConnectionError",
    "OpenSearchClient",
    "OpenSearchClientError",
]


def __getattr__(name: str):
    if name in {"DatabaseClient", "DatabaseConnectionError"}:
        from .database import DatabaseClient, DatabaseConnectionError

        return {
            "DatabaseClient": DatabaseClient,
            "DatabaseConnectionError": DatabaseConnectionError,
        }[name]

    if name in {"OpenSearchClient", "OpenSearchClientError"}:
        from .opensearch import OpenSearchClient, OpenSearchClientError

        return {
            "OpenSearchClient": OpenSearchClient,
            "OpenSearchClientError": OpenSearchClientError,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

