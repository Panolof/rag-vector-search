"""Compatibility entry point for one-off searches."""

import sys

from rag_vector_search.cli import main as cli_main


def process_query(query: str) -> int:
    return cli_main(["search", query])


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit("Usage: python src/rag_pipeline.py <question>")
    raise SystemExit(process_query(question))
