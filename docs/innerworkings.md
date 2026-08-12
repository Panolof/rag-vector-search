# How the demo works

The default path is local and inspectable:

```text
checked-in synthetic records
          |
          v
deterministic feature-hash embeddings
          |
          v
Milvus Lite collection on a local .db file
          |
          v
cosine retrieval for the user's question
          |
          v
score and shared-term relevance gate
          |
          +--> visible evidence cards
          |
          v
extractive answer, or an explicitly selected provider adapter
```

## Corpus

`src/rag_vector_search/resources/sample_news.jsonl` contains 18 original synthetic records. The default demo downloads no dataset and redistributes no third-party article text.

`corpus.py` validates required fields, positive unique IDs, non-empty values, and text limits. A SHA-256 fingerprint covers the canonical records plus the embedding contract.

## Embeddings

`HashingEmbedder` tokenises lower-case text, removes a small stop-word set, normalises common plurals, adds adjacent-token features, hashes each feature into a fixed-size vector, and applies L2 normalisation.

The same object embeds documents and questions. It needs no model artefact or network request. It is a deterministic lexical baseline, not a claim of state-of-the-art semantic retrieval.

## Milvus storage

`MilvusStore` uses the current `MilvusClient` API against a local Milvus Lite file by default. Its schema keeps the full evidence needed by the UI:

- stable integer ID;
- 384-dimensional vector;
- category;
- title;
- summary;
- source label;
- corpus and embedding fingerprint.

The vector index uses `AUTOINDEX` with cosine similarity.

Bootstrap is non-destructive. If the named collection exists, the code loads it and verifies its schema, dimension, row count, IDs, and fingerprint. A mismatch does not drop or overwrite data. It asks the operator to choose a new versioned collection name.

## Retrieval and generation

`RagPipeline` limits and normalises the question. Milvus returns a larger nearest-neighbour candidate set. `MilvusStore` keeps at most `RAG_TOP_K` records that meet `RAG_MIN_SCORE` and share at least `RAG_MIN_SHARED_TERMS` distinct normalised non-stopword terms with the question. This deterministic guard filters the tested no-overlap, hash-collision, and one-term-bait cases. It is a lexical heuristic, not an entailment check. The pipeline validates each accepted entity and gives those exact records to a generator.

The default `ExtractiveGenerator` produces a predictable answer from the best matches. The optional `OpenAIGenerator` is lazy-loaded only when `RAG_GENERATOR=openai`. It requires an explicit model name, bounds context and output size, and tells the model to treat retrieved text as untrusted data rather than instructions.

No provider call runs during import, bootstrap, testing, or the default interface.

## Web boundary

The Flask app uses an application factory. Importing it starts no database, network listener, model, or provider request. A pipeline is created only for a submitted search unless the CLI injects one at startup.

The app limits request and question size, relies on Jinja autoescaping, and returns a restrictive Content Security Policy plus anti-framing, no-sniff, and no-referrer headers. Flask debug mode stays off.

The built-in server is for local demonstration only. The web factory accepts only the free extractive generator; optional paid-provider mode is terminal-only, so a cross-origin form cannot trigger provider calls against localhost. The form has no CSRF token because the local web path has no authentication, paid call, or state-changing operation beyond creating the synthetic index. Put a production WSGI server and normal deployment controls in front of the application for shared use. Milvus Lite also permits only one process to own a local database file, so stop the server before opening the same file from the CLI.

## Trust boundaries

1. Corpus rows are data. Validate them before indexing.
2. Retrieved rows are untrusted context. Escape them in HTML and do not execute their instructions.
3. Environment variables may contain secrets. Never print or commit them.
4. A remote Milvus URI or provider mode introduces network and credential boundaries absent from the local demo.
5. Dependency audits detect known reports. They do not prove a package harmless.
