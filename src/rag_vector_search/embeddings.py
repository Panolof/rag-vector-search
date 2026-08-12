from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class HashingEmbedder:
    """Create deterministic, local embeddings with signed feature hashing.

    This intentionally small baseline needs no model download. It is suitable for
    a reproducible demo, not for production semantic retrieval.
    """

    dimension: int = 384
    name: str = "hashing-v2"

    def __post_init__(self) -> None:
        if self.dimension < 8:
            raise ValueError("dimension must be at least 8")

    def embed(self, text: str) -> list[float]:
        tokens = list(self.terms(text))
        if not tokens:
            return [0.0] * self.dimension

        features = tokens + [f"{left}::{right}" for left, right in zip(tokens, tokens[1:])]
        counts = Counter(features)
        vector = [0.0] * self.dimension

        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def contract(self) -> str:
        """Fingerprint observable embedding behaviour, not only its name."""
        probe = self.embed("dependency reviews and local vector search")
        payload = ",".join(f"{value:.12g}" for value in probe)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.name}:{self.dimension}:{digest}"

    def terms(self, text: str) -> tuple[str, ...]:
        """Return the lexical features used by the visible relevance guard."""
        return tuple(
            _normalise_token(token)
            for token in _TOKEN.findall(text.lower())
            if token not in _STOP_WORDS
        )


def _normalise_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
