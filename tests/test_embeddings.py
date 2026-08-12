import math

from rag_vector_search.embeddings import HashingEmbedder


def test_embeddings_are_deterministic_and_normalised():
    embedder = HashingEmbedder(64)

    first = embedder.embed("dependency security review")
    second = embedder.embed("dependency security review")

    assert first == second
    assert len(first) == 64
    assert math.isclose(sum(value * value for value in first), 1.0, rel_tol=1e-7)


def test_empty_text_returns_zero_vector():
    assert HashingEmbedder(16).embed(" -- ") == [0.0] * 16


def test_terms_normalise_simple_plurals_for_the_relevance_guard():
    assert HashingEmbedder().terms("Batteries and dependencies") == (
        "battery",
        "dependency",
    )


def test_embedding_contract_covers_observable_output(monkeypatch):
    embedder = HashingEmbedder(64)
    original = embedder.contract()

    monkeypatch.setattr(HashingEmbedder, "embed", lambda self, _text: [1.0] * self.dimension)

    assert embedder.contract() != original
