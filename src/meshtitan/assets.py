from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import AssetRecord

_ASSET_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class AssetError(ValueError):
    pass


class AssetStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, data: bytes, original_name: str, media_type: str | None = None) -> AssetRecord:
        suffix = Path(original_name).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise AssetError(f"unsupported image extension {suffix!r}; expected one of {sorted(_ALLOWED_SUFFIXES)}")
        digest = hashlib.sha256(data).hexdigest()
        directory = self.root / digest
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"input{suffix}"
        if not image_path.exists():
            image_path.write_bytes(data)
        record = AssetRecord(
            asset_id=digest,
            original_name=Path(original_name).name,
            media_type=media_type,
            bytes=len(data),
            sha256=digest,
            path=str(image_path.resolve()),
        )
        (directory / "asset.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return record

    def get(self, asset_id: str) -> AssetRecord:
        if not _ASSET_RE.fullmatch(asset_id):
            raise AssetError("invalid asset id")
        meta = self.root / asset_id / "asset.json"
        if not meta.is_file():
            raise AssetError(f"asset not found: {asset_id}")
        return AssetRecord.model_validate(json.loads(meta.read_text(encoding="utf-8")))
