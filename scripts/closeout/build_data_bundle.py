"""Deterministic private transfer of a sealed prepared dataset, never Git input."""

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path

from flir_pipeline.closeout.io import verify


def build(source, output):
    verify(source)
    if output.exists() or source.name != "flir-denoised-exploratory-20261122-data":
        raise ValueError("Expected fresh private exploratory transfer artifact")
    with (
        output.open("xb") as stream,
        gzip.GzipFile(
            fileobj=stream, mode="wb", filename="", mtime=0, compresslevel=1
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = tarfile.TarInfo(
                source.name + "/" + path.relative_to(source).as_posix()
            )
            info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
            with path.open("rb") as content:
                archive.addfile(info, content)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(f"{digest}  {output.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
