from pathlib import Path

from contract_rag.data.download import CUAD_JSON_NAME, download_cuad


def test_download_skips_when_json_exists(tmp_path: Path) -> None:
    existing = tmp_path / CUAD_JSON_NAME
    existing.write_text("{}", encoding="utf-8")

    # No network access happens: the function returns the existing file immediately
    assert download_cuad(tmp_path) == existing
