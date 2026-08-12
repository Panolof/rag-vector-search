import pytest

from rag_vector_search.config import Settings


def test_default_settings_are_local_and_offline(monkeypatch):
    for name in (
        "RAG_DB_URI",
        "RAG_COLLECTION",
        "RAG_DIMENSION",
        "RAG_TOP_K",
        "RAG_MIN_SCORE",
        "RAG_MIN_SHARED_TERMS",
        "RAG_GENERATOR",
        "MILVUS_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_environment()

    assert settings.db_uri == "data/news_demo.db"
    assert settings.collection == "synthetic_news_v4"
    assert settings.generator == "extractive"
    assert settings.min_score == 0.08
    assert settings.min_shared_terms == 2
    assert settings.milvus_token is None


@pytest.mark.parametrize("name", ["bad-name", "9bad", "spaces are bad"])
def test_collection_name_is_restricted(name):
    with pytest.raises(ValueError, match="RAG_COLLECTION"):
        Settings(collection=name).validated()


def test_invalid_integer_environment_fails_clearly(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "many")

    with pytest.raises(ValueError, match="RAG_TOP_K must be an integer"):
        Settings.from_environment()


def test_invalid_score_environment_fails_clearly(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SCORE", "1.1")

    with pytest.raises(ValueError, match="RAG_MIN_SCORE must be between 0 and 1"):
        Settings.from_environment()


@pytest.mark.parametrize(
    "uri",
    ["http://milvus.example", "tcp://milvus.example:19530", "data/news.db"],
)
def test_remote_token_requires_https(uri):
    with pytest.raises(ValueError, match="requires an https"):
        Settings(
            db_uri=uri,
            milvus_token="not-a-real-token",
        ).validated()


def test_remote_token_accepts_https():
    assert Settings(
        db_uri="https://milvus.example",
        milvus_token="not-a-real-token",
    ).validated()


@pytest.mark.parametrize(
    "uri",
    [
        "https://user:password@milvus.example",
        "https://user@milvus.example",
        "https://:password@milvus.example",
        "user:password@milvus.example:19530",
    ],
)
def test_remote_uri_rejects_embedded_credentials(uri):
    with pytest.raises(ValueError, match="must not contain embedded credentials"):
        Settings(db_uri=uri).validated()
