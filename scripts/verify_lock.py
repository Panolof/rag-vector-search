#!/usr/bin/env python3
"""Fail closed unless every locked registry package passes two online checks.

This script uses only the Python standard library. Run it before `uv sync`.
It verifies lockfile provenance and hashes locally, then checks each exact
package release against PyPI's JSON API and OSV's batch API.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


PYPI_INDEX = "https://pypi.org/simple"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
USER_AGENT = "rag-vector-search-lock-verifier/0.2"


@dataclass(frozen=True, order=True)
class LockedRelease:
    name: str
    version: str
    hashes: frozenset[str]


def load_locked_releases(lock_path: Path) -> list[LockedRelease]:
    with lock_path.open("rb") as handle:
        lock = tomllib.load(handle)

    releases: list[LockedRelease] = []
    for package in lock.get("package", []):
        source = package.get("source", {})
        if "editable" in source:
            if source != {"editable": "."}:
                raise ValueError(f"unexpected editable source for {package['name']}: {source}")
            continue
        if source != {"registry": PYPI_INDEX}:
            raise ValueError(f"non-PyPI source for {package['name']}: {source}")

        artifacts = []
        if package.get("sdist"):
            artifacts.append(package["sdist"])
        artifacts.extend(package.get("wheels", []))
        if not artifacts:
            raise ValueError(f"no locked artifacts for {package['name']}=={package['version']}")

        hashes: set[str] = set()
        for artifact in artifacts:
            digest = artifact.get("hash", "")
            if not digest.startswith("sha256:") or len(digest) != 71:
                raise ValueError(
                    f"invalid artifact hash for {package['name']}=={package['version']}"
                )
            if not str(artifact.get("url", "")).startswith("https://files.pythonhosted.org/"):
                raise ValueError(
                    f"unexpected artifact host for {package['name']}=={package['version']}"
                )
            hashes.add(digest.removeprefix("sha256:"))

        releases.append(
            LockedRelease(
                name=str(package["name"]),
                version=str(package["version"]),
                hashes=frozenset(hashes),
            )
        )
    if not releases:
        raise ValueError("lockfile contains no registry packages")
    return sorted(set(releases))


def verify_pypi_release(release: LockedRelease) -> tuple[LockedRelease, list[dict]]:
    name = urllib.parse.quote(release.name, safe="")
    version = urllib.parse.quote(release.version, safe="")
    payload = request_json(PYPI_JSON.format(name=name, version=version))
    info = payload.get("info", {})
    if info.get("version") != release.version:
        raise ValueError(f"PyPI version mismatch for {release.name}=={release.version}")
    if info.get("yanked"):
        raise ValueError(f"PyPI release is yanked: {release.name}=={release.version}")

    registry_files = {
        str(item.get("digests", {}).get("sha256", "")): item
        for item in payload.get("urls", [])
        if item.get("digests", {}).get("sha256")
    }
    registry_hashes = set(registry_files)
    missing = release.hashes.difference(registry_hashes)
    if missing:
        raise ValueError(
            f"lockfile hash is absent from PyPI for {release.name}=={release.version}"
        )
    if any(registry_files[digest].get("yanked") for digest in release.hashes):
        raise ValueError(f"locked PyPI artifact is yanked: {release.name}=={release.version}")
    vulnerabilities = payload.get("vulnerabilities", [])
    if not isinstance(vulnerabilities, list):
        raise ValueError(f"invalid PyPI vulnerability response for {release.name}")
    return release, vulnerabilities


def query_osv(releases: list[LockedRelease]) -> list[list[dict]]:
    body = {
        "queries": [
            {
                "package": {"ecosystem": "PyPI", "name": release.name},
                "version": release.version,
            }
            for release in releases
        ]
    }
    payload = request_json(OSV_BATCH, body=body)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(releases):
        raise ValueError("OSV returned an incomplete batch response")
    findings: list[list[dict]] = []
    for release, result in zip(releases, results, strict=True):
        if not isinstance(result, dict):
            raise ValueError(f"OSV returned an invalid result for {release.name}")
        vulnerabilities = result.get("vulns", [])
        if not isinstance(vulnerabilities, list):
            raise ValueError(f"OSV returned invalid advisories for {release.name}")
        findings.append(vulnerabilities)
    return findings


def request_json(url: str, *, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"{url} returned a non-object JSON response")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="verify sources and hashes without making network requests",
    )
    args = parser.parse_args(argv)

    try:
        releases = load_locked_releases(args.lock)
        print(f"Lock structure: PASS ({len(releases)} exact PyPI releases, hashed artifacts only)")
        if args.structure_only:
            return 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            pypi_results = list(executor.map(verify_pypi_release, releases))
        pypi_findings = [
            (release, finding)
            for release, findings in pypi_results
            for finding in findings
        ]
        osv_results = query_osv(releases)
        osv_findings = [
            (release, finding)
            for release, findings in zip(releases, osv_results, strict=True)
            for finding in findings
        ]
        if pypi_findings or osv_findings:
            print("Vulnerability verification: FAIL", file=sys.stderr)
            for source, findings in (("PyPI", pypi_findings), ("OSV", osv_findings)):
                for release, finding in findings:
                    advisory = finding.get("id", "unknown-advisory")
                    print(
                        f"- {source}: {release.name}=={release.version}: {advisory}",
                        file=sys.stderr,
                    )
            return 1
        print(
            f"Online vulnerability verification: PASS ({len(releases)} releases, "
            "PyPI and OSV returned no known advisories)"
        )
        return 0
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Lock verification: FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
