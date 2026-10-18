"""Download selected inference weights at immutable revisions; no GPU is used."""
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
snapshot_download(
    "facebook/dinov2-large", revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    allow_patterns=["config.json"],
)
snapshot_download(
    "tencent/Hunyuan3D-2mv", revision="3a761b539b29fe4ff64714813aa9560fd66f5de0",
    allow_patterns=["hunyuan3d-dit-v2-mv/config.yaml", "hunyuan3d-dit-v2-mv/model.fp16.safetensors"],
    local_dir=Path.home() / ".cache/hy3dgen/tencent/Hunyuan3D-2mv",
)
snapshot_download(
    "infinith/UltraShape", revision="5aeb21a7185d39f042d02b2695802f125a6f5159",
    allow_patterns=["ultrashape_v1.pt"], local_dir=root / "var/models/UltraShape",
)
