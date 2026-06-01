import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline.manifest_manager import ManifestManager


@pytest.fixture
def manifest_manager(tmp_path):
    with patch("src.pipeline.manifest_manager.PROCESSED_DATA_DIR", tmp_path):
        manager = ManifestManager(parser_type="manual")
        return manager


def test_update_parser_type(manifest_manager):
    assert manifest_manager.parser_type == "manual"
    manifest_manager.update_parser_type("docling")
    assert manifest_manager.parser_type == "docling"


def test_make_manifest(manifest_manager):
    files = {"test.pdf": {"hash": "123", "parser_type": "manual"}}
    manifest = manifest_manager.make_manifest(files)
    assert manifest["version"] == "2.0"
    assert manifest["global_parser_type"] == "manual"
    assert manifest["files"] == files


def test_load_manifest_not_exists(manifest_manager):
    manifest = manifest_manager.load_manifest()
    assert manifest["version"] == "2.0"
    assert manifest["files"] == {}


def test_load_manifest_valid_2_0(manifest_manager, tmp_path):
    valid_data = {"version": "2.0", "global_parser_type": "manual", "files": {"a.pdf": {"hash": "abc"}}}
    with open(manifest_manager.manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_data, f)

    manifest = manifest_manager.load_manifest()
    assert manifest["version"] == "2.0"
    assert "a.pdf" in manifest["files"]


def test_load_manifest_legacy_1_0(manifest_manager, tmp_path):
    legacy_data = {"a.pdf": "old_hash"}
    with open(manifest_manager.manifest_path, "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)

    manifest = manifest_manager.load_manifest()
    assert manifest["version"] == "2.0"
    assert manifest["files"]["a.pdf"]["hash"] == "old_hash"
    assert manifest["files"]["a.pdf"]["parser_type"] == "manual"


def test_load_manifest_corrupted(manifest_manager, tmp_path):
    with open(manifest_manager.manifest_path, "w", encoding="utf-8") as f:
        f.write("{invalid_json}")

    manifest = manifest_manager.load_manifest()
    assert manifest["version"] == "2.0"
    assert manifest["files"] == {}
    assert not manifest_manager.manifest_path.exists()


def test_save_manifest(manifest_manager):
    data = {"version": "2.0", "global_parser_type": "manual", "files": {}}
    manifest_manager.save_manifest(data)

    assert manifest_manager.manifest_path.exists()
    with open(manifest_manager.manifest_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_find_relative_path(manifest_manager):
    old_files = {"folder/test1.pdf": {}, "test2.pdf": {}}
    scanned_files = [Path("RAW/folder2/test3.pdf")]

    # 1. Exact match in old_files
    assert manifest_manager.find_relative_path("folder/test1.pdf", old_files, scanned_files) == "folder/test1.pdf"

    # 2. Match by name in old_files
    assert manifest_manager.find_relative_path("test2.pdf", old_files, scanned_files) == "test2.pdf"

    # 3. Match in scanned_files
    with patch("src.pipeline.manifest_manager.RAW_DATA_DIR", Path("RAW")):
        assert manifest_manager.find_relative_path("test3.pdf", old_files, scanned_files) == "folder2/test3.pdf"

    # 4. Not found
    assert manifest_manager.find_relative_path("unknown.pdf", old_files, scanned_files) is None
