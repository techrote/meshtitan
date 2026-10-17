from pathlib import Path

from meshtitan.config import load_config, resolve_python


def test_config_resolves_paths_relative_to_file(tmp_path: Path):
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    py = tmp_path / "venv" / "bin" / "python"
    py.write_text("", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    config = tmp_path / "meshtitan.toml"
    config.write_text(
        """
[runtime]
data_dir = "./state"
[gpu]
selector = "GPU-test"
[backends.triposr]
python = "./venv"
repo = "./repo"
commit = "abc"
""",
        encoding="utf-8",
    )

    parsed = load_config(config)
    assert parsed.data_dir == (tmp_path / "state").resolve()
    assert parsed.backends["triposr"].repo == (tmp_path / "repo").resolve()
    assert resolve_python(parsed.backends["triposr"].python) == py
