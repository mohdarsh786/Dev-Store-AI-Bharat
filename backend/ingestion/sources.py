"""Source adapters that reuse the existing fetchers and emit canonical resources."""

from __future__ import annotations

from typing import Dict, List, Tuple

from config import settings
from models import CanonicalResource, IngestionSource

from .fetchers.github_fetcher import GitHubFetcher
from .fetchers.huggingface_fetcher import HuggingFaceFetcher
from .fetchers.kaggle_fetcher import KaggleFetcher
from .fetchers.openrouter_fetcher import OpenRouterFetcher
from .normalization import canonicalize_resource


class GitHubSourceAdapter:
    source = IngestionSource.GITHUB

    def __init__(self, fetcher: GitHubFetcher | None = None):
        self.fetcher = fetcher or GitHubFetcher(token=settings.ingestion_github_api_token)

    def fetch(self) -> Tuple[List[dict], List[CanonicalResource]]:
        raw_payloads = self.fetcher.fetch_all_repositories()
        resources = []
        for raw in raw_payloads:
            normalized = self.fetcher.normalize_repository(raw)
            if normalized:
                resources.append(canonicalize_resource(self.source, normalized, raw))
        return raw_payloads, resources


class HuggingFaceSourceAdapter:
    source = IngestionSource.HUGGINGFACE

    def __init__(self, fetcher: HuggingFaceFetcher | None = None):
        self.fetcher = fetcher or HuggingFaceFetcher()

    def fetch(self) -> Tuple[List[dict], List[CanonicalResource]]:
        raw_models = self.fetcher.fetch_all_models()
        raw_datasets = self.fetcher.fetch_datasets()
        resources: List[CanonicalResource] = []
        for raw in raw_models:
            normalized = self.fetcher.normalize_model(raw)
            if normalized:
                resources.append(canonicalize_resource(self.source, normalized, raw))
        for raw in raw_datasets:
            normalized = self.fetcher.normalize_dataset(raw)
            if normalized:
                resources.append(canonicalize_resource(self.source, normalized, raw))
        return raw_models + raw_datasets, resources


class KaggleSourceAdapter:
    source = IngestionSource.KAGGLE

    def __init__(self, fetcher: KaggleFetcher | None = None, max_pages: int = 5, page_size: int = 100):
        self.fetcher = fetcher or KaggleFetcher()
        self.max_pages = max_pages
        self.page_size = page_size

    def fetch(self) -> Tuple[List[dict], List[CanonicalResource]]:
        raw_payloads: List[dict] = []
        for page in range(1, self.max_pages + 1):
            datasets = self.fetcher.fetch_datasets(page=page, page_size=self.page_size)
            if not datasets:
                break
            raw_payloads.extend(datasets)

        resources = []
        for raw in raw_payloads:
            normalized = self.fetcher.normalize_dataset(raw)
            if normalized:
                resources.append(canonicalize_resource(self.source, normalized, raw))
        return raw_payloads, resources


class OpenRouterSourceAdapter:
    source = IngestionSource.OPENROUTER

    def __init__(self, fetcher: OpenRouterFetcher | None = None):
        self.fetcher = fetcher or OpenRouterFetcher()

    def fetch(self) -> Tuple[List[dict], List[CanonicalResource]]:
        raw_payloads = self.fetcher.fetch_models()
        resources = []
        for raw in raw_payloads:
            normalized = self.fetcher.normalize_model(raw)
            if normalized:
                resources.append(canonicalize_resource(self.source, normalized, raw))
        return raw_payloads, resources


def build_default_source_adapters() -> Dict[str, object]:
    return {
        IngestionSource.GITHUB.value: GitHubSourceAdapter(),
        IngestionSource.HUGGINGFACE.value: HuggingFaceSourceAdapter(),
        IngestionSource.KAGGLE.value: KaggleSourceAdapter(),
        IngestionSource.OPENROUTER.value: OpenRouterSourceAdapter(),
    }
