import pytest

from rag_vector_search.config import MAX_QUERY_CHARS
from rag_vector_search.pipeline import RagPipeline


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, query, *, limit):
        self.calls.append((query, limit))
        return [
            {
                "id": 4,
                "distance": 0.875,
                "entity": {
                    "category": "Cyber security",
                    "title": "Dependency reviews shift earlier",
                    "summary": "Teams check packages before installation.",
                    "source": "synthetic-demo",
                },
            }
        ]


def test_pipeline_returns_answer_and_visible_evidence():
    store = FakeStore()
    pipeline = RagPipeline(store, top_k=3)

    result = pipeline.query("  dependency   security  ")

    assert result.query == "dependency security"
    assert result.hits[0].id == 4
    assert result.hits[0].score == 0.875
    assert "Dependency reviews" in result.answer
    assert store.calls == [("dependency security", 3)]


@pytest.mark.parametrize("query", ["", "   "])
def test_pipeline_rejects_blank_questions(query):
    with pytest.raises(ValueError, match="Enter a question"):
        RagPipeline(FakeStore()).query(query)


def test_pipeline_rejects_oversized_questions():
    with pytest.raises(ValueError, match="characters or fewer"):
        RagPipeline(FakeStore()).query("x" * (MAX_QUERY_CHARS + 1))
