import json

import pytest

from rag_vector_search.corpus import load_corpus


def test_packaged_corpus_is_synthetic_and_unique():
    documents = load_corpus()

    assert len(documents) == 18
    assert len({document.id for document in documents}) == len(documents)
    assert {document.source for document in documents} == {"synthetic-demo"}


def test_corpus_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = {
        "id": 1,
        "category": "Test",
        "title": "Synthetic title",
        "summary": "Synthetic summary",
        "source": "synthetic-demo",
    }
    path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id 1"):
        load_corpus(path)
