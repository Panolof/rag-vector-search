"""Compatibility wrapper for the old Milvus setup command."""

from rag_vector_search.cli import main as cli_main


def main():
    return cli_main(["bootstrap"])


if __name__ == "__main__":
    raise SystemExit(main())
