from dataclasses import replace

import pytest

from rag_vector_search.corpus import load_corpus
from rag_vector_search.embeddings import HashingEmbedder
from rag_vector_search.store import CollectionMismatchError, MilvusStore


def make_store(tmp_path):
    return MilvusStore(
        uri=str(tmp_path / "news.db"),
        collection="test_news_v1",
        embedder=HashingEmbedder(384),
    )


def test_milvus_lite_bootstrap_is_idempotent_and_searches(tmp_path):
    store = make_store(tmp_path)
    documents = load_corpus()

    assert store.bootstrap(documents) == 18
    assert store.bootstrap(documents) == 18
    hits = store.search("What is changing in software dependency security?", limit=3)

    assert hits
    assert hits[0]["id"] == 4
    assert hits[0]["entity"]["source"] == "synthetic-demo"
    store.client.close()


@pytest.mark.parametrize(
    "query",
    [
        "How do I bake a chocolate cake?",
        "What is the capital of Peru?",
        "How do teams bake a chocolate cake?",
        "How can batteries cure a headache?",
        "team astrology",
    ],
)
def test_out_of_domain_queries_do_not_return_arbitrary_neighbours(tmp_path, query):
    store = make_store(tmp_path)
    store.bootstrap(load_corpus())

    assert store.search(query, limit=4) == []
    store.client.close()


def test_existing_collection_mismatch_preserves_data(tmp_path):
    store = make_store(tmp_path)
    documents = load_corpus()
    store.bootstrap(documents)
    changed = [replace(documents[0], summary="Changed synthetic content"), *documents[1:]]

    with pytest.raises(CollectionMismatchError, match="Choose a new RAG_COLLECTION"):
        store.bootstrap(changed)

    assert int(store.client.get_collection_stats(store.collection)["row_count"]) == 18
    store.client.close()


def test_new_client_reopens_and_loads_existing_collection(tmp_path):
    first = make_store(tmp_path)
    documents = load_corpus()
    first.bootstrap(documents)
    first.client.close()

    reopened = make_store(tmp_path)
    assert reopened.bootstrap(documents) == 18
    assert reopened.search("local vector database", limit=1)
    reopened.client.close()
