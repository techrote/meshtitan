from pathlib import Path

import pytest

from meshtitan.assets import AssetError, AssetStore


def test_assets_are_content_addressed(tmp_path: Path):
    store = AssetStore(tmp_path)
    first = store.add(b"abc", "input.png", "image/png")
    second = store.add(b"abc", "different.png", "image/png")
    assert first.asset_id == second.asset_id
    assert store.get(first.asset_id).sha256 == first.sha256


def test_assets_reject_non_image_suffix(tmp_path: Path):
    store = AssetStore(tmp_path)
    with pytest.raises(AssetError):
        store.add(b"abc", "payload.exe")
