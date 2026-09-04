from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


IGNORED_PARTS = {".git", "runtime", "__pycache__"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_manifest(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash a small explicit trust boundary; missing files remain observable."""
    result: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = root / relative
        result[relative] = sha256_file(path) if path.is_file() else "<missing>"
    return result


def assert_protected_unchanged(
    root: Path, expected: dict[str, str], *, context: str,
) -> None:
    observed = protected_manifest(root, expected)
    changed = sorted(
        path for path, digest in expected.items() if observed.get(path) != digest
    )
    if changed:
        raise RuntimeError(
            f"protected files changed during {context}: " + ", ".join(changed)
        )


def tree_manifest(root: Path) -> dict[str, str]:
    """Hash a lightweight Agent snapshot for strict write-scope verification."""
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        result[relative.as_posix()] = sha256_file(path)
    return result


def assert_changes_within(
    before: dict[str, str], after: dict[str, str], *,
    allowed_prefixes: Iterable[str], context: str,
) -> list[str]:
    allowed = tuple(prefix.rstrip("/") + "/" for prefix in allowed_prefixes)
    changed = sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    escaped = [path for path in changed if not path.startswith(allowed)]
    if escaped:
        raise RuntimeError(
            f"Agent wrote outside its assigned version during {context}: "
            + ", ".join(escaped)
        )
    return changed


def verify_dataset_manifest(
    root: Path, manifest_path: Path, *, deep: bool = False,
) -> tuple[bool, str]:
    """Validate sizes cheaply; optionally verify the multi-gigabyte file hashes."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            path = manifest_path.parent / name
            if not path.is_file():
                return False, f"missing dataset file: {path}"
            if path.stat().st_size != int(expected["bytes"]):
                return False, f"dataset size mismatch: {path}"
            if deep and sha256_file(path) != expected["sha256"]:
                return False, f"dataset SHA-256 mismatch: {path}"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"invalid dataset manifest: {error}"
    mode = "size+sha256" if deep else "size"
    return True, f"{mode} verified"


def dataset_file_signatures(manifest_path: Path) -> dict[str, tuple[int, int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        name: (path.stat().st_size, path.stat().st_mtime_ns)
        for name in manifest["files"]
        for path in [manifest_path.parent / name]
    }


def assert_dataset_signatures(
    manifest_path: Path, expected: dict[str, tuple[int, int]], *, context: str,
) -> None:
    observed = dataset_file_signatures(manifest_path)
    if observed != expected:
        changed = sorted(set(expected) | set(observed))
        raise RuntimeError(
            f"dataset files changed during {context}: " + ", ".join(changed)
        )
