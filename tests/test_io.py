import json

from semcom.utils.config import load_config
from semcom.utils.io import ensure_dir, save_config, save_json


def test_save_config(tmp_path) -> None:
    cfg = load_config()
    output_dir = ensure_dir(tmp_path / "debug_output")

    save_config(cfg, output_dir)

    assert (output_dir / "compiled_config.yaml").exists()


def test_save_json_replaces_empty_directory_path(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.mkdir()

    save_json({"loss": 3.25}, path)

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {"loss": 3.25}


def test_save_json(tmp_path) -> None:
    output_path = tmp_path / "metrics.json"

    save_json(
        data={"loss": 4.2, "accuracy": 0.75},
        path=output_path,
    )

    assert output_path.exists()
    assert "loss" in output_path.read_text(encoding="utf-8")
