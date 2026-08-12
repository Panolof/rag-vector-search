from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class NewsDocument:
    id: int
    category: str
    title: str
    summary: str
    source: str

    @property
    def text(self) -> str:
        return f"{self.category}. {self.title}. {self.summary}"

    @classmethod
    def from_mapping(cls, row: dict[str, object], *, line_number: int) -> "NewsDocument":
        required = {"id", "category", "title", "summary", "source"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")

        document = cls(
            id=int(row["id"]),
            category=str(row["category"]).strip(),
            title=str(row["title"]).strip(),
            summary=str(row["summary"]).strip(),
            source=str(row["source"]).strip(),
        )
        if document.id < 1:
            raise ValueError(f"line {line_number}: id must be positive")
        for name in ("category", "title", "summary", "source"):
            if not getattr(document, name):
                raise ValueError(f"line {line_number}: {name} must not be empty")
        if len(document.title) > 240 or len(document.summary) > 2_000:
            raise ValueError(f"line {line_number}: document text exceeds the demo limit")
        return document


def load_corpus(path: Path | None = None) -> list[NewsDocument]:
    documents: list[NewsDocument] = []
    seen_ids: set[int] = set()

    corpus_path = path or files("rag_vector_search").joinpath("resources/sample_news.jsonl")
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            document = NewsDocument.from_mapping(row, line_number=line_number)
            if document.id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {document.id}")
            seen_ids.add(document.id)
            documents.append(document)

    if not documents:
        raise ValueError(f"corpus is empty: {corpus_path}")
    return documents


def document_rows(documents: Iterable[NewsDocument]) -> list[dict[str, object]]:
    return [
        {
            "id": document.id,
            "category": document.category,
            "title": document.title,
            "summary": document.summary,
            "source": document.source,
        }
        for document in documents
    ]
