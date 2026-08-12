# Third-party notices

## Milvus standalone Docker Compose files in Git history

Reachable Git history contains earlier `docker-compose.yml` files obtained from
official Milvus release assets. They are not used by the current v0.2 demo. The
current Compose file is an independently written, empty migration marker.

- Commit `f0c0f062f5f3` contains the unmodified Milvus v2.0.2 standalone
  Compose release asset.
- Commit `f9bb9b03427d` contains the unmodified Milvus v2.4.0 standalone
  Compose release asset.
- Commit `836828c9865b` contains the Milvus v2.4.6 standalone Compose release
  asset with two MinIO root-credential environment fields added locally.

Official release assets:

- <https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml>
- <https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml>
- <https://github.com/milvus-io/milvus/releases/download/v2.4.6/milvus-standalone-docker-compose.yml>

Milvus is distributed under the Apache License 2.0. The historical upstream
files and the stated local modification remain subject to that licence. A copy
is provided at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

The Milvus name is used only to identify the origin of the historical files.
No endorsement is implied.
