"""Preserve an operator's config and GPU selection; add missing extension tables."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "meshtitan.toml"
if path.exists():
    text = path.read_text(encoding="utf-8")
else:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader",
    ], text=True).splitlines()
    titans = [row.split(",", 1)[0].strip() for row in rows if "TITAN RTX" in row]
    if len(titans) != 1:
        raise SystemExit("Expected exactly one TITAN RTX; create meshtitan.toml and select its UUID explicitly.")
    text = (root / "meshtitan.toml.example").read_text(encoding="utf-8")
    text = re.sub(r'^selector = .*$', f'selector = "{titans[0]}"', text, flags=re.M)

tables = {
    "hunyuan_multiview": '''enabled = true
python = ".venv-hunyuan-mv"
repo = ".vendor/Hunyuan3D-2"
commit = "f8db63096c8282cb27354314d896feba5ba6ff8a"
model_path = "tencent/Hunyuan3D-2mv"
subfolder = "hunyuan3d-dit-v2-mv"
model_revision = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
''',
    "ultrashape": '''enabled = true
python = ".venv-ultrashape"
repo = ".vendor/UltraShape-1.0"
commit = "5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66"
checkpoint = "../../var/models/UltraShape/ultrashape_v1.pt"
model_revision = "5aeb21a7185d39f042d02b2695802f125a6f5159"
''',
}
for name, table in tables.items():
    if f"[backends.{name}]" not in text:
        text += f"\n[backends.{name}]\n{table}"
# Existing example files already contain extension tables; never overwrite custom settings.
path.write_text(text, encoding="utf-8")
print(json.dumps({"configured": str(path), "added_backends": list(tables)}))
