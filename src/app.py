"""Compatibility entry point for the original `python src/app.py` command."""

from rag_vector_search.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
