"""Local-first retrieval-augmented generation with Milvus Lite."""

from .pipeline import RagPipeline, RagResponse, SearchHit

__all__ = ["RagPipeline", "RagResponse", "SearchHit"]
__version__ = "0.2.0"
