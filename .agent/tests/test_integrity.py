from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))
from core.integrity import (  # noqa: E402
    assert_changes_within,
    assert_protected_unchanged,
    protected_manifest,
    tree_manifest,
    verify_dataset_manifest,
)


class IntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hif4_integrity_")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_protected_change_is_detected(self) -> None:
        path = self.root / "evaluator.py"
        path.write_text("one", encoding="utf-8")
        expected = protected_manifest(self.root, ["evaluator.py"])
        path.write_text("two", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            assert_protected_unchanged(self.root, expected, context="test")

    def test_agent_may_only_change_assigned_version(self) -> None:
        target = self.root / "solution" / "v1_method"
        target.mkdir(parents=True)
        file = target / "solution.py"
        file.write_text("old", encoding="utf-8")
        other = self.root / "README.md"
        other.write_text("old", encoding="utf-8")
        before = tree_manifest(self.root)
        file.write_text("new", encoding="utf-8")
        after = tree_manifest(self.root)
        self.assertEqual(
            assert_changes_within(
                before, after, allowed_prefixes=["solution/v1_method"], context="test"
            ),
            ["solution/v1_method/solution.py"],
        )
        other.write_text("new", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            assert_changes_within(
                before, tree_manifest(self.root),
                allowed_prefixes=["solution/v1_method"], context="test",
            )

    def test_dataset_manifest_supports_fast_and_deep_checks(self) -> None:
        data = self.root / "data"
        data.mkdir()
        payload = b"representative"
        (data / "linear.pt").write_bytes(payload)
        manifest = {
            "files": {
                "linear.pt": {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            }
        }
        path = data / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(verify_dataset_manifest(self.root, path)[0])
        self.assertTrue(verify_dataset_manifest(self.root, path, deep=True)[0])


if __name__ == "__main__":
    unittest.main()
