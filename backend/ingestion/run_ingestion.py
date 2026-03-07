"""CLI entrypoint for the production ingestion orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.orchestrator import IngestionOrchestrator


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(message)s",
    )


async def _run(args: argparse.Namespace) -> int:
    orchestrator = IngestionOrchestrator()
    result = await orchestrator.run(
        sources=args.sources,
        persist_snapshots=not args.no_snapshots,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"success", "partial_success"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DevStore production ingestion")
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=["github", "huggingface", "kaggle", "openrouter"],
        help="Run only the selected ingestion sources",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Skip S3 raw snapshot persistence",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
