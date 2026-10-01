from pathlib import Path

from contract_rag.data.download import CUAD_JSON_NAME, TEST_JSON_NAME, download_cuad


def test_download_skips_when_files_exist(tmp_path: Path) -> None:
    (tmp_path / CUAD_JSON_NAME).write_text("{}", encoding="utf-8")
    (tmp_path / TEST_JSON_NAME).write_text("{}", encoding="utf-8")

    assert download_cuad(tmp_path) == tmp_path / CUAD_JSON_NAME
