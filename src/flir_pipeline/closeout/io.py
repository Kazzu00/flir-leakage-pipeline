"""Immutable output transactions and portable byte receipts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from flir_pipeline.similarity.storage import file_sha256


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_relative(name: str) -> str:
    p = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or p.is_absolute()
        or ".." in p.parts
        or p.as_posix() != name
        or any(part.endswith((".", " ")) for part in p.parts)
        or any(ord(c) < 32 for c in name)
        or any(
            part.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"COM{i}" for i in range(1, 10)],
                *[f"LPT{i}" for i in range(1, 10)],
            }
            for part in p.parts
        )
    ):
        raise ValueError(f"Unsafe relative path: {name!r}")
    return name


def environment() -> dict:
    root = Path(__file__).resolve().parents[3]
    commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        pass
    bundle = root / "BUNDLE_MANIFEST.json"
    if commit is None and bundle.is_file():
        commit = read_json(bundle).get("git_commit")
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "git_commit": commit,
    }


@contextmanager
def transaction(output: Path, *sources: Path):
    """Publish once by sibling rename; retain failed staging as explicit evidence."""
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output must be disjoint from input sources")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        yield stage
        stage.rename(output)
    except BaseException as exc:
        write_json(
            stage / "FAILURE.json",
            {
                "state": "FAILED_NOT_PUBLISHED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def seal(root: Path, state: dict) -> dict:
    result = {
        **state,
        "files": {
            p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p != root / "receipt.json"
        },
    }
    write_json(root / "receipt.json", result)
    return result


def verify(root: Path) -> dict:
    receipt = read_json(root / "receipt.json")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "receipt.json"
    }
    if actual != set(receipt["files"]):
        raise ValueError("Receipt file inventory changed")
    for name, sha in receipt["files"].items():
        safe_relative(name)
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or file_sha256(p) != sha:
            raise ValueError(f"Receipt checksum mismatch: {name}")
    return receipt
