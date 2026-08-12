# Security policy

## Supported version

Security fixes target the latest revision of `main`.

## Report a vulnerability

Please use GitHub's private vulnerability reporting feature for this repository. Do not place credentials, exploit payloads, or private data in a public issue.

## Dependency policy

- Runtime and development dependencies are locked in `uv.lock` with SHA-256 artefact hashes.
- `python3 scripts/verify_lock.py` checks every exact release against PyPI and OSV before installation.
- `pip-audit` checks the installed environment after installation.
- GitHub Actions use full commit identifiers and read-only token permissions.
- No vulnerability exceptions are configured. A positive finding fails the check.

These checks cover known public advisories. They do not prove that a package is free of malicious or unknown behaviour. Review provenance and minimise dependencies before adding a package.
