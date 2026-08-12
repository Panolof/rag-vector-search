"""Compatibility wrapper for the old preprocessing command.

The v0.2 demo embeds its checked-in synthetic corpus during bootstrap.
"""

from rag_vector_search.cli import main as cli_main

def main():
    return cli_main(["bootstrap"])

if __name__ == "__main__":
    raise SystemExit(main())
