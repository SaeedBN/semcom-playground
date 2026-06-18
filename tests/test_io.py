from semcom.utils.config import load_config
from semcom.utils.io import ensure_dict, save_config


def test_save_config(tmp_path) -> None:
    cfg = load_config()
    output_dir = ensure_dict(tmp_path / "debug_output")

    save_config(cfg, output_dir)

    assert (output_dir / "compiled_config.yaml").exists()
