import os
import subprocess
import sys


def test_importing_application_creates_no_database_or_network_service(tmp_path):
    environment = os.environ.copy()
    environment["RAG_DB_URI"] = str(tmp_path / "should-not-exist.db")

    result = subprocess.run(
        [sys.executable, "-c", "import rag_vector_search.app"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "should-not-exist.db").exists()
