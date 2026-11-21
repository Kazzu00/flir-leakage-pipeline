"""Standalone standard-library integrity verifier; never imports GPU packages."""

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


def verify(root):
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        p = root / name
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not p.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Unsafe bundle path")
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    print(f"VERIFIED {len(manifest['files'])} code/config/documentation files")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
