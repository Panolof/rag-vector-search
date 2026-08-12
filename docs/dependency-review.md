# Dependency review

Reviewed: 2026-08-12

## Gate

No package was installed until these checks passed:

1. Confirm the direct package, current release, owner, Python support, and release hashes on its official PyPI page.
2. Generate `uv.lock` without installing project packages.
3. Run `uv run --no-project --python 3.11 python scripts/verify_lock.py`.
4. Install only from the unchanged lock with `uv sync --locked`.
5. Run `uv run --frozen pip-audit --local` against the installed environment.

The pre-install verifier checks every locked registry release, not only direct dependencies. It accepts only PyPI sources and `files.pythonhosted.org` artefacts, compares every locked SHA-256 digest with PyPI, rejects yanked releases, and asks both PyPI and OSV for known advisories. Network or incomplete-response failures are blocking.

## Direct packages

| Package | Locked version | Role | Official evidence |
|---|---:|---|---|
| Flask | 3.1.3 | Local web interface | [PyPI](https://pypi.org/project/Flask/3.1.3/) |
| PyMilvus | 3.0.1 | Milvus client | [PyPI](https://pypi.org/project/pymilvus/3.0.1/) |
| Milvus Lite | 3.1.0 | Embedded vector database | [PyPI](https://pypi.org/project/milvus-lite/3.1.0/) |
| OpenAI | 2.46.0 | Optional provider adapter | [PyPI](https://pypi.org/project/openai/2.46.0/) |
| pytest | 9.1.1 | Test runner | [PyPI](https://pypi.org/project/pytest/9.1.1/) |
| pip-audit | 2.10.1 | Installed-environment audit | [PyPI](https://pypi.org/project/pip-audit/2.10.1/) |
| Hatchling | 1.28.0 | Build backend | [PyPI](https://pypi.org/project/hatchling/1.28.0/) |

The PyPI JSON records for these exact releases listed no known vulnerabilities at review time. `scripts/verify_lock.py` extended that check across all 71 exact cross-platform releases in the lock and OSV returned no known advisories.

CI uses `uv` 0.12.0 through the immutable `setup-uv` v9.0.0 action. It uses `actions/checkout` v7.0.1, whose safer fork defaults and dependency updates are relevant to untrusted pull requests. Both actions are pinned to their full, verified release commit identifiers. The exact `uv` release had 19 non-yanked PyPI artefacts and no PyPI or OSV advisories at review time.

This is a dated observation, not a permanent safety claim. Re-run both checks before each dependency change or release.
