import pytest
from types import SimpleNamespace

from rag_vector_search.app import create_app
from rag_vector_search import app as app_module
from rag_vector_search.config import Settings
from rag_vector_search.generation import OpenAIGenerator
from rag_vector_search.pipeline import RagPipeline
from rag_vector_search.store import CollectionMismatchError


class FakeStore:
    def search(self, query, *, limit):
        del query, limit
        return [
            {
                "id": 7,
                "distance": 0.75,
                "entity": {
                    "category": "Climate",
                    "title": "<script>alert('x')</script>",
                    "summary": "Synthetic summary",
                    "source": "synthetic-demo",
                },
            }
        ]


def make_client():
    app = create_app(RagPipeline(FakeStore()))
    app.config.update(TESTING=True)
    return app.test_client()


def test_get_renders_real_interface_and_security_headers():
    response = make_client().get("/")

    assert response.status_code == 200
    assert b"Ask the corpus" in response.data
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_post_renders_answer_sources_and_escapes_content():
    response = make_client().post("/", data={"query": "city heat"})

    assert response.status_code == 200
    assert b"RETRIEVED EVIDENCE" in response.data
    assert b"&lt;script&gt;" in response.data
    assert b"<script>alert" not in response.data


def test_blank_query_returns_user_error():
    response = make_client().post("/", data={"query": " "})

    assert response.status_code == 400
    assert b"Enter a question" in response.data


def test_web_factory_rejects_provider_settings():
    with pytest.raises(ValueError, match="extractive generation only"):
        create_app(settings=Settings(generator="openai"))


def test_web_factory_validates_explicit_settings():
    with pytest.raises(ValueError, match="top_k"):
        create_app(settings=Settings(top_k=0))


def test_web_factory_rejects_injected_provider_pipeline():
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: None))
    pipeline = RagPipeline(
        FakeStore(),
        generator=OpenAIGenerator(model="test-model", client=client),
    )

    with pytest.raises(ValueError, match="extractive generation only"):
        create_app(pipeline)


def test_injected_pipeline_ignores_unrelated_environment(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "many")

    assert create_app(RagPipeline(FakeStore())).test_client().get("/").status_code == 200


def test_lazy_pipeline_is_built_once_under_parallel_first_requests(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time

    builds = 0
    builds_lock = Lock()

    def build_once(_settings):
        nonlocal builds
        with builds_lock:
            builds += 1
        time.sleep(0.05)
        return RagPipeline(FakeStore())

    monkeypatch.setattr(app_module, "_build_pipeline", build_once)
    application = create_app()
    application.config.update(TESTING=True)

    def post_query(_index):
        with application.test_client() as client:
            return client.post("/", data={"query": "city heat"}).status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(post_query, range(4)))

    assert statuses == [200, 200, 200, 200]
    assert builds == 1


def test_example_buttons_bypass_empty_search_field_validation():
    response = make_client().get("/")

    assert response.data.count(b"formnovalidate") == 3


def test_collection_mismatch_reaches_the_user():
    class MismatchedStore:
        def search(self, _query, *, limit):
            del limit
            raise CollectionMismatchError("Choose a new RAG_COLLECTION")

    app = create_app(RagPipeline(MismatchedStore()))
    app.config.update(TESTING=True)
    response = app.test_client().post("/", data={"query": "city heat"})

    assert response.status_code == 409
    assert b"Choose a new RAG_COLLECTION" in response.data
