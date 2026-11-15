"""Read-only integrity, frozen-source and replay checks for the closure package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from flir_pipeline.general.publication import read_json, write_json
from flir_pipeline.utils.hashing import sha256_file


def check(repo: Path, data: Path, prior: Path, package: Path):
    output = package / "verification"
    output.mkdir(exist_ok=False)
    old = read_json(prior / "previous_publications_sha256.json")
    changed = [
        name
        for name, expected in old.items()
        if sha256_file(data / "operations" / name) != expected
    ]
    release = read_json(prior / "release_receipt.json")["files_sha256"]
    changed_release = [
        name
        for name, expected in release.items()
        if sha256_file(prior / name) != expected
    ]
    snapshot = prior / "implementation-final"
    source_hashes = {
        p.relative_to(snapshot).as_posix(): sha256_file(p)
        for p in (snapshot / "src").rglob("*.py")
    }
    changed_source = [
        name
        for name, expected in source_hashes.items()
        if not (repo / name).exists() or sha256_file(repo / name) != expected
    ]
    added_source = sorted(
        {p.relative_to(repo).as_posix() for p in (repo / "src").rglob("*.py")}
        - source_hashes.keys()
    )
    integrity = {
        "prior_publication_files": len(old),
        "changed_prior_publications": changed,
        "prior_release_files": len(release),
        "changed_prior_release": changed_release,
        "frozen_source_files": len(source_hashes),
        "changed_source": changed_source,
        "added_source": added_source,
        "frozen_source_sha256": source_hashes,
    }
    write_json(output / "integrity.json", integrity)
    if changed or changed_release or changed_source or added_source:
        raise ValueError("Integrity/freeze comparison failed; inspect evidence")
    env = {
        **os.environ,
        **{
            k: "1"
            for k in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "OMP_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    commands = []

    def run(name, argv):
        with (output / f"{name}.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                argv,
                cwd=repo,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        commands.append({"name": name, "argv": argv, "exit_code": result.returncode})
        write_json(output / "commands.json", commands)
        print(f"{name}: {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed")

    run(
        "pertinent-tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_general_pipeline.py",
            "tests/test_splitting.py",
            "tests/test_paired_delivery.py",
            "tests/test_identity.py",
            "tests/test_manifest.py",
            "tests/test_visual_localization.py",
            "-q",
            f"--junitxml={output / 'pertinent-tests.xml'}",
        ],
    )
    run("ruff", [sys.executable, "-m", "ruff", "check", "."])
    run("diff-check", ["git", "diff", "--check"])
    cli = [sys.executable, "-c", "from flir_pipeline.cli import app; app()", "general"]
    run("cli-help", [*cli, "--help"])
    for case in read_json(package / "benchmark/protocol.json")["cases"]:
        root = package / "benchmark" / case
        run(
            f"replay-{case}",
            [
                *cli,
                "verify",
                str(root / "publication"),
                str(root / "inputs"),
                "--report",
                str(output / f"replay-{case}.json"),
            ],
        )
    run(
        "replay-fac",
        [
            *cli,
            "verify",
            str(prior / "fac-final"),
            str(data),
            "--report",
            str(output / "replay-fac.json"),
        ],
    )
    write_json(
        output / "completed.json",
        {
            "verified": True,
            "global_ci_green": False,
            "full_global_suite_rerun": False,
            "scope": "77 pertinent tests, independent replays and immutable input/source checks; organization comparison is recorded separately",
        },
    )
    print(
        json.dumps({k: v for k, v in integrity.items() if k != "frozen_source_sha256"})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "data", "prior", "package"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    check(*(p.resolve() for p in (args.repo, args.data, args.prior, args.package)))
