from __future__ import annotations

from pathlib import Path
from threading import Lock

from flask import Flask, render_template, request

from .config import MAX_QUERY_CHARS, Settings
from .pipeline import RagPipeline
from .store import CollectionMismatchError


def create_app(
    pipeline: RagPipeline | None = None,
    *,
    settings: Settings | None = None,
) -> Flask:
    template_dir, static_dir = _asset_paths()
    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.config.update(MAX_CONTENT_LENGTH=16 * 1024)
    active_settings = (
        settings.validated()
        if settings is not None
        else (Settings() if pipeline is not None else Settings.from_environment())
    )
    if active_settings.generator != "extractive":
        raise ValueError("The web demo supports extractive generation only")
    if pipeline is not None and pipeline.generator.name != "extractive":
        raise ValueError("The web demo supports extractive generation only")
    app.extensions["rag_pipeline"] = pipeline
    app.extensions["rag_settings"] = active_settings
    app.extensions["rag_pipeline_lock"] = Lock()

    @app.after_request
    def add_security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; img-src 'self' data:; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return render_template("index.html", max_query_chars=MAX_QUERY_CHARS)

        query_values = request.form.getlist("query")
        query = next((value for value in reversed(query_values) if value.strip()), "")
        try:
            active_pipeline = app.extensions.get("rag_pipeline")
            if active_pipeline is None:
                with app.extensions["rag_pipeline_lock"]:
                    active_pipeline = app.extensions.get("rag_pipeline")
                    if active_pipeline is None:
                        active_pipeline = _build_pipeline(app.extensions["rag_settings"])
                        app.extensions["rag_pipeline"] = active_pipeline
            result = active_pipeline.query(query)
        except ValueError as exc:
            return (
                render_template(
                    "index.html",
                    error=str(exc),
                    query=query[:MAX_QUERY_CHARS],
                    max_query_chars=MAX_QUERY_CHARS,
                ),
                400,
            )
        except CollectionMismatchError as exc:
            return (
                render_template(
                    "index.html",
                    error=str(exc),
                    query=query[:MAX_QUERY_CHARS],
                    max_query_chars=MAX_QUERY_CHARS,
                ),
                409,
            )
        return render_template(
            "index.html",
            result=result,
            query=result.query,
            max_query_chars=MAX_QUERY_CHARS,
        )

    return app


def _build_pipeline(settings: Settings) -> RagPipeline:
    from .cli import build_pipeline

    return build_pipeline(settings, bootstrap=True)


def _asset_paths() -> tuple[Path, Path]:
    package_dir = Path(__file__).resolve().parent
    packaged_templates = package_dir / "templates"
    packaged_static = package_dir / "static"
    if packaged_templates.exists() and packaged_static.exists():
        return packaged_templates, packaged_static

    repository_root = package_dir.parents[1]
    template_dir = repository_root / "templates"
    static_dir = repository_root / "static"
    if not (repository_root / "pyproject.toml").exists():
        return packaged_templates, packaged_static
    return template_dir, static_dir
