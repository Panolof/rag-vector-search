from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import MAX_QUERY_CHARS
from .generation import ExtractiveGenerator, Generator


class VectorSearch(Protocol):
    def search(self, query: str, *, limit: int) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class SearchHit:
    id: int
    score: float
    category: str
    title: str
    summary: str
    source: str


@dataclass(frozen=True)
class RagResponse:
    query: str
    answer: str
    hits: tuple[SearchHit, ...]
    generator: str


class RagPipeline:
    def __init__(
        self,
        store: VectorSearch,
        *,
        generator: Generator | None = None,
        top_k: int = 4,
    ) -> None:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        self.store = store
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k

    def query(self, query: str) -> RagResponse:
        cleaned = " ".join(query.split())
        if not cleaned:
            raise ValueError("Enter a question to search the corpus")
        if len(cleaned) > MAX_QUERY_CHARS:
            raise ValueError(f"Question must be {MAX_QUERY_CHARS} characters or fewer")

        hits = tuple(self._parse_hit(hit) for hit in self.store.search(cleaned, limit=self.top_k))
        answer = self.generator.answer(cleaned, hits)
        return RagResponse(
            query=cleaned,
            answer=answer,
            hits=hits,
            generator=self.generator.name,
        )

    @staticmethod
    def _parse_hit(raw: dict[str, object]) -> SearchHit:
        entity = raw.get("entity")
        if not isinstance(entity, dict):
            raise RuntimeError("Milvus returned a hit without an entity payload")
        return SearchHit(
            id=int(raw["id"]),
            score=float(raw["distance"]),
            category=str(entity["category"]),
            title=str(entity["title"]),
            summary=str(entity["summary"]),
            source=str(entity["source"]),
        )
