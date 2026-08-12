from pathlib import Path

import pytest

from scripts.verify_lock import LockedRelease, load_locked_releases, verify_pypi_release


def test_lock_uses_only_hashed_pypi_artifacts():
    releases = load_locked_releases(Path("uv.lock"))

    assert len(releases) >= 3
    assert all(release.hashes for release in releases)
    assert {release.name for release in releases}.issuperset(
        {"flask", "milvus-lite", "pymilvus", "pytest"}
    )


def test_lock_verifier_rejects_git_sources(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        """
version = 1
[[package]]
name = "unsafe"
version = "1.0.0"
source = { git = "https://example.invalid/repo" }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-PyPI source"):
        load_locked_releases(lock)


def test_pypi_verifier_rejects_a_yanked_locked_artifact(monkeypatch):
    release = LockedRelease("demo", "1.2.3", frozenset({"a" * 64}))

    monkeypatch.setattr(
        "scripts.verify_lock.request_json",
        lambda _url: {
            "info": {"version": "1.2.3", "yanked": False},
            "urls": [
                {
                    "digests": {"sha256": "a" * 64},
                    "yanked": True,
                    "yanked_reason": "broken build",
                }
            ],
            "vulnerabilities": [],
        },
    )

    with pytest.raises(ValueError, match="locked PyPI artifact is yanked"):
        verify_pypi_release(release)


def test_pypi_verifier_rejects_a_hash_absent_from_pypi(monkeypatch):
    release = LockedRelease("demo", "1.2.3", frozenset({"a" * 64}))
    monkeypatch.setattr(
        "scripts.verify_lock.request_json",
        lambda _url: {
            "info": {"version": "1.2.3", "yanked": False},
            "urls": [{"digests": {"sha256": "b" * 64}, "yanked": False}],
            "vulnerabilities": [],
        },
    )

    with pytest.raises(ValueError, match="hash is absent"):
        verify_pypi_release(release)
