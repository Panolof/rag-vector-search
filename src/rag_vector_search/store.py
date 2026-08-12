from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pymilvus import DataType, MilvusClient

from .corpus import NewsDocument
from .embeddings import HashingEmbedder


VECTOR_FIELD = "embedding"
OUTPUT_FIELDS = ["category", "title", "summary", "source", "corpus_hash"]
FIELD_CONTRACT = {
    "id": (DataType.INT64, None),
    VECTOR_FIELD: (DataType.FLOAT_VECTOR, None),
    "category": (DataType.VARCHAR, 80),
    "title": (DataType.VARCHAR, 512),
    "summary": (DataType.VARCHAR, 4096),
    "source": (DataType.VARCHAR, 160),
    "corpus_hash": (DataType.VARCHAR, 64),
}


class CollectionMismatchError(RuntimeError):
    """Raised when a collection exists but does not match this demo's contract."""


class MilvusStore:
    def __init__(
        self,
        *,
        uri: str,
        collection: str,
        embedder: HashingEmbedder,
        min_score: float = 0.08,
        min_shared_terms: int = 2,
        token: str | None = None,
        client: MilvusClient | None = None,
    ) -> None:
        self.uri = uri
        self.collection = collection
        self.embedder = embedder
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        self.min_score = min_score
        if min_shared_terms < 1:
            raise ValueError("min_shared_terms must be positive")
        self.min_shared_terms = min_shared_terms
        self._prepare_local_path(uri)
        kwargs: dict[str, str] = {"uri": uri}
        if token:
            kwargs["token"] = token
        self.client = client or MilvusClient(**kwargs)

    def bootstrap(self, documents: Iterable[NewsDocument]) -> int:
        docs = list(documents)
        if not docs:
            raise ValueError("cannot bootstrap an empty corpus")
        corpus_hash = corpus_fingerprint(
            docs,
            embedding_contract=self.embedder.contract(),
        )

        if self.client.has_collection(collection_name=self.collection):
            self._validate_existing_collection(docs, corpus_hash)
            return len(docs)

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name=VECTOR_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedder.dimension,
        )
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=80)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=160)
        schema.add_field(field_name="corpus_hash", datatype=DataType.VARCHAR, max_length=64)

        index = self.client.prepare_index_params()
        index.add_index(
            field_name=VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )

        vectors = self.embedder.embed_many([document.text for document in docs])
        rows = [
            {
                "id": document.id,
                VECTOR_FIELD: vector,
                "category": document.category,
                "title": document.title,
                "summary": document.summary,
                "source": document.source,
                "corpus_hash": corpus_hash,
            }
            for document, vector in zip(docs, vectors, strict=True)
        ]
        result = self.client.insert(collection_name=self.collection, data=rows)
        inserted = int(result.get("insert_count", 0))
        if inserted != len(rows):
            raise RuntimeError(f"Milvus inserted {inserted} of {len(rows)} documents")
        return inserted

    def search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        vector = self.embedder.embed(query)
        if not any(vector):
            return []
        query_terms = set(self.embedder.terms(query))
        result_sets = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field=VECTOR_FIELD,
            limit=max(limit * 4, limit),
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        candidates = result_sets[0] if result_sets else []
        matches: list[dict[str, object]] = []
        for candidate in candidates:
            score = float(candidate["distance"])
            entity = candidate.get("entity")
            if not isinstance(entity, dict):
                raise RuntimeError("Milvus returned a hit without an entity payload")
            candidate_text = " ".join(
                str(entity[field]) for field in ("category", "title", "summary")
            )
            candidate_terms = set(self.embedder.terms(candidate_text))
            shared_terms = query_terms.intersection(candidate_terms)
            if score < self.min_score or len(shared_terms) < self.min_shared_terms:
                continue
            matches.append(candidate)
            if len(matches) == limit:
                break
        return matches

    def _validate_existing_collection(
        self,
        documents: list[NewsDocument],
        corpus_hash: str,
    ) -> None:
        self.client.load_collection(collection_name=self.collection)
        description = self.client.describe_collection(collection_name=self.collection)
        fields = {field["name"]: field for field in description.get("fields", [])}
        required = {"id", VECTOR_FIELD, *OUTPUT_FIELDS}
        if set(fields) != required:
            missing = required.difference(fields)
            extra = set(fields).difference(required)
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unexpected fields: {', '.join(sorted(extra))}")
            self._mismatch("; ".join(details))

        if description.get("auto_id") or description.get("enable_dynamic_field"):
            self._mismatch("automatic IDs or dynamic fields are enabled")
        if description.get("consistency_level_name") != "Strong":
            self._mismatch("consistency level is not Strong")

        for field_name, (expected_type, max_length) in FIELD_CONTRACT.items():
            field = fields[field_name]
            if field.get("type") != expected_type:
                self._mismatch(f"field '{field_name}' has the wrong type")
            if max_length is not None:
                observed_length = int(field.get("params", {}).get("max_length", 0))
                if observed_length != max_length:
                    self._mismatch(
                        f"field '{field_name}' max_length is {observed_length}; "
                        f"expected {max_length}"
                    )
        if not fields["id"].get("is_primary"):
            self._mismatch("field 'id' is not the primary key")

        dimension = int(fields[VECTOR_FIELD].get("params", {}).get("dim", 0))
        if dimension != self.embedder.dimension:
            self._mismatch(
                f"embedding dimension is {dimension}; expected {self.embedder.dimension}"
            )

        index = self.client.describe_index(
            collection_name=self.collection,
            index_name=VECTOR_FIELD,
        )
        if index.get("field_name") != VECTOR_FIELD:
            self._mismatch("vector index targets the wrong field")
        if index.get("index_type") != "AUTOINDEX" or index.get("metric_type") != "COSINE":
            self._mismatch("vector index must use AUTOINDEX with COSINE")

        stats = self.client.get_collection_stats(collection_name=self.collection)
        row_count = int(stats.get("row_count", 0))
        if row_count != len(documents):
            self._mismatch(f"row count is {row_count}; expected {len(documents)}")

        stored = self.client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["id", "corpus_hash"],
            limit=max(1, len(documents)),
        )
        expected_ids = {document.id for document in documents}
        stored_ids = {int(row["id"]) for row in stored}
        stored_hashes = {str(row["corpus_hash"]) for row in stored}
        if stored_ids != expected_ids or stored_hashes != {corpus_hash}:
            self._mismatch("stored IDs or corpus fingerprint differ")

    def _mismatch(self, reason: str) -> None:
        raise CollectionMismatchError(
            f"Collection '{self.collection}' is incompatible: {reason}. "
            "Choose a new RAG_COLLECTION name to preserve the existing data."
        )

    @staticmethod
    def _prepare_local_path(uri: str) -> None:
        if "://" in uri:
            return
        path = Path(uri).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)


def corpus_fingerprint(
    documents: Iterable[NewsDocument],
    *,
    embedding_contract: str,
) -> str:
    canonical = [
        {
            "id": document.id,
            "category": document.category,
            "title": document.title,
            "summary": document.summary,
            "source": document.source,
        }
        for document in documents
    ]
    payload = json.dumps(
        {"documents": canonical, "embedding_contract": embedding_contract},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
