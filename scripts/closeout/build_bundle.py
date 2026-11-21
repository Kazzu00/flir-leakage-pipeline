"""Deterministic code-only deployment archive, with strict explicit path families."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def build(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        text=True,
    ).strip():
        raise ValueError("Deployment bundle requires a committed, clean code checkout")
    allowed = [
        root / "src",
        root / "tests/closeout",
        root / "configs",
        root / "scripts/closeout",
    ]
    paths = {
        p
        for folder in allowed
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".yaml", ".json", ".html")
        and "__pycache__" not in p.parts
    }
    paths.update(
        root / name
        for name in (
            "README.md",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "docs/runbooks/oe4_closeout_20261120.md",
        )
    )
    paths.update(p for p in (root / "docs").rglob("*.md") if p.is_file())
    paths.update(p for p in (root / "scripts/hypatia").glob("*oe4*") if p.is_file())
    paths.update(
        root / name
        for name in (
            "scripts/hypatia/denoised_closeout.sbatch",
            "scripts/hypatia/deploy_to_hypatia.ps1",
        )
    )
    blobs = {
        p.relative_to(root).as_posix(): p.read_bytes().replace(b"\r\n", b"\n")
        for p in sorted(paths)
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    manifest = {
        "schema": "flir_code_bundle_v1",
        "git_commit": commit,
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()},
        "contains_real_data": False,
        "contains_checkpoints": False,
        "included_tests": "tests/closeout; full repository suite is verified separately",
    }
    blobs["BUNDLE_MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for name, content in sorted(blobs.items()):
            info = tarfile.TarInfo("flir-oe4-closeout-20261121-code/" + name)
            info.size = len(content)
            info.mode = 0o755 if name.endswith((".sh", ".sbatch")) else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha}  {output.name}\n", encoding="ascii"
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"bundle": str(output), "sha256": sha, "files": len(blobs)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[2], args.output)
