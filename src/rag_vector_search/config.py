from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit


DEFAULT_DB_URI = "data/news_demo.db"
DEFAULT_COLLECTION = "synthetic_news_v4"
DEFAULT_DIMENSION = 384
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.08
DEFAULT_MIN_SHARED_TERMS = 2
MAX_QUERY_CHARS = 300

_COLLECTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


@dataclass(frozen=True)
class Settings:
    db_uri: str = DEFAULT_DB_URI
    collection: str = DEFAULT_COLLECTION
    dimension: int = DEFAULT_DIMENSION
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    min_shared_terms: int = DEFAULT_MIN_SHARED_TERMS
    generator: str = "extractive"
    milvus_token: str | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            db_uri=os.getenv("RAG_DB_URI", DEFAULT_DB_URI),
            collection=os.getenv("RAG_COLLECTION", DEFAULT_COLLECTION),
            dimension=_positive_int("RAG_DIMENSION", DEFAULT_DIMENSION, maximum=4096),
            top_k=_positive_int("RAG_TOP_K", DEFAULT_TOP_K, maximum=20),
            min_score=_bounded_float("RAG_MIN_SCORE", DEFAULT_MIN_SCORE),
            min_shared_terms=_positive_int(
                "RAG_MIN_SHARED_TERMS",
                DEFAULT_MIN_SHARED_TERMS,
                maximum=20,
            ),
            generator=os.getenv("RAG_GENERATOR", "extractive").strip().lower(),
            milvus_token=os.getenv("MILVUS_TOKEN") or None,
        ).validated()

    def validated(self) -> "Settings":
        if not self.db_uri.strip():
            raise ValueError("RAG_DB_URI must not be empty")
        if "://" in self.db_uri:
            try:
                parsed_uri = urlsplit(self.db_uri)
            except ValueError as exc:
                raise ValueError("RAG_DB_URI is not a valid remote URI") from exc
            if parsed_uri.username is not None or parsed_uri.password is not None:
                raise ValueError(
                    "RAG_DB_URI must not contain embedded credentials; "
                    "use MILVUS_TOKEN"
                )
        elif "@" in self.db_uri:
            raise ValueError(
                "RAG_DB_URI must not contain embedded credentials; "
                "use MILVUS_TOKEN"
            )
        if not _COLLECTION_NAME.fullmatch(self.collection):
            raise ValueError(
                "RAG_COLLECTION must start with a letter or underscore and contain "
                "only letters, numbers, and underscores"
            )
        if not 1 <= self.dimension <= 4096:
            raise ValueError("dimension must be between 1 and 4096")
        if not 1 <= self.top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if not 1 <= self.min_shared_terms <= 20:
            raise ValueError("min_shared_terms must be between 1 and 20")
        if self.generator not in {"extractive", "openai"}:
            raise ValueError("RAG_GENERATOR must be 'extractive' or 'openai'")
        if self.milvus_token and not self.db_uri.lower().startswith("https://"):
            raise ValueError("MILVUS_TOKEN requires an https:// RAG_DB_URI")
        return self


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _bounded_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value
