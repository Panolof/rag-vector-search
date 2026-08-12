from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import Settings
from .corpus import load_corpus
from .embeddings import HashingEmbedder
from .generation import build_generator
from .pipeline import RagPipeline
from .store import MilvusStore


def build_pipeline(settings: Settings, *, bootstrap: bool) -> RagPipeline:
    embedder = HashingEmbedder(settings.dimension)
    store = MilvusStore(
        uri=settings.db_uri,
        collection=settings.collection,
        embedder=embedder,
        min_score=settings.min_score,
        min_shared_terms=settings.min_shared_terms,
        token=settings.milvus_token,
    )
    if bootstrap:
        store.bootstrap(load_corpus())
    return RagPipeline(
        store,
        generator=build_generator(settings.generator),
        top_k=settings.top_k,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-vector-search",
        description="Local-first Milvus Lite retrieval demo",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("bootstrap", help="Create or verify the synthetic demo index")

    search = subcommands.add_parser("search", help="Search the demo corpus")
    search.add_argument("query", help="Question to search for")

    serve = subcommands.add_parser("serve", help="Run the local Flask interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=5000, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_environment()

    if args.command == "bootstrap":
        pipeline = build_pipeline(settings, bootstrap=True)
        del pipeline
        print(
            f"Ready: collection '{settings.collection}' at {settings.db_uri} "
            f"with {len(load_corpus())} synthetic records."
        )
        return 0

    if args.command == "search":
        result = build_pipeline(settings, bootstrap=True).query(args.query)
        print(json.dumps(asdict(result), indent=2))
        return 0

    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if settings.generator != "extractive":
        raise SystemExit(
            "Provider generation is CLI-only. Set RAG_GENERATOR=extractive "
            "before starting the web demo."
        )
    from .app import create_app

    app = create_app(build_pipeline(settings, bootstrap=True), settings=settings)
    app.run(host=args.host, port=args.port, debug=False)
    return 0
