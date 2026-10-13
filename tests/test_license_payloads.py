"""Reject missing upstream terms or newly unaccounted bundled dependencies."""
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from license_check import check_licenses


def source_copy(tmp_path):
    (tmp_path / 'tools').mkdir()
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'Cargo.lock', 'tools/notice_manifest.json'):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def test_complete_locked_license_content():
    check_licenses(ROOT)


def test_removed_disclaimer_is_rejected(tmp_path):
    root = source_copy(tmp_path)
    notices = root / 'THIRD_PARTY_NOTICES.md'
    original = notices.read_text()
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in original
    notices.write_text(original.replace('THE SOFTWARE IS PROVIDED "AS IS"', '', 1))
    with pytest.raises(ValueError, match='Incomplete or altered upstream terms'):
        check_licenses(root)


def test_unaccounted_cargo_version_is_rejected(tmp_path):
    root = source_copy(tmp_path)
    path = root / 'tools/notice_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cargo_components'].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dependency versions'):
        check_licenses(root)
