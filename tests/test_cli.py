import pytest

from rag_vector_search import cli
from rag_vector_search.config import Settings


def test_provider_mode_refuses_web_server(monkeypatch):
    monkeypatch.setattr(
        cli.Settings,
        "from_environment",
        lambda: Settings(generator="openai"),
    )

    with pytest.raises(SystemExit, match="CLI-only"):
        cli.main(["serve"])


def test_cli_rejects_embedded_uri_credentials_without_echoing_them(
    monkeypatch,
    capsys,
):
    password = "demo-password"
    monkeypatch.setenv(
        "RAG_DB_URI",
        f"https://demo-user:{password}@milvus.example",
    )

    with pytest.raises(ValueError, match="must not contain embedded credentials"):
        cli.main(["bootstrap"])

    captured = capsys.readouterr()
    assert password not in captured.out
    assert password not in captured.err
