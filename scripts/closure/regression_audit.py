"""Frozen-code regression comparison; no product code or test semantics changed.

Both arms run in their own source root, with the same Python/environment/tests.
The previous arm restores the committed MILP and saved pre-general CLI.
Pre-existing local random allocation and organization changes remain in both arms.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def save(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def outcomes(path):
    result = {}
    for case in ET.parse(path).iter("testcase"):
        fail = case.find("failure")
        error = case.find("error")
        result[case.attrib["name"]] = {
            "state": "failed"
            if fail is not None
            else "error"
            if error is not None
            else "passed",
            "message": fail.attrib.get("message", "") if fail is not None else "",
            "trace": fail.text if fail is not None else "",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.repo, args.prior, args.output = (
        p.resolve() for p in (args.repo, args.prior, args.output)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep nested scientific fixture paths below Windows' path-length ceiling.
    temp_root = Path(
        tempfile.mkdtemp(prefix="flir-freeze-tests-", dir=args.output.anchor)
    )
    env = dict(os.environ)
    env.update(
        {
            k: "1"
            for k in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "OMP_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        }
    )
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "controls": {
            k: env[k] for k in env if k.endswith("NUM_THREADS") or k.endswith("OFFLINE")
        },
    }
    save(args.output / "environment.json", environment)
    for arm in ("previous", "candidate"):
        root = args.output / arm
        root.mkdir()
        for directory in ("src", "tests", "exports/frontend/organization/schema"):
            shutil.copytree(
                args.repo / directory,
                root / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for name in ("pyproject.toml", "uv.lock", ".gitignore"):
            shutil.copy2(args.repo / name, root / name)
        if arm == "previous":
            shutil.copy2(
                args.prior / "baseline-regression/src/flir_pipeline/cli.py",
                root / "src/flir_pipeline/cli.py",
            )
            # The old saved control accidentally contains the new MILP. Obtain
            # the actual previous function from Git and preserve the unrelated
            # local atomic_random_assignment that predates general integration.
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
            ).strip()
            committed = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{commit}:src/flir_pipeline/splitting/construction.py",
                ],
                cwd=args.repo,
                text=True,
            )
            target = root / "src/flir_pipeline/splitting/construction.py"
            current = target.read_text(encoding="utf-8")
            prefix = current.split("def milp_assignment(", 1)[0]
            error_start, record_start = (
                prefix.index("class AssignmentError("),
                prefix.index("def record_statistics("),
            )
            prefix = prefix[:error_start] + prefix[record_start:]
            target.write_text(
                prefix
                + "def milp_assignment("
                + committed.split("def milp_assignment(", 1)[1],
                encoding="utf-8",
            )
            save(
                args.output / "previous-construction-origin.json",
                {
                    "commit": commit,
                    "path": "src/flir_pipeline/splitting/construction.py",
                    "restored": "committed milp_assignment and following code; no AssignmentError",
                    "preserved": "prior local atomic_random_assignment and organization contract",
                    "saved_prior_solver_control_rejected": "identical to candidate; inadequate solver control",
                },
            )
        arm_env = {**env, "PYTHONPATH": str(root / "src")}
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_organization_export.py",
            "-q",
            "--tb=short",
            f"--junitxml={args.output / (arm + '.xml')}",
            f"--basetemp={temp_root / arm}",
        ]
        save(
            args.output / f"{arm}-command.json",
            {"cwd": str(root), "argv": command, "PYTHONPATH": arm_env["PYTHONPATH"]},
        )
        with (args.output / f"{arm}.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=root,
                env=arm_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"{arm} exit={completed.returncode}", flush=True)
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"{arm} did not complete pytest")
    before, after = (
        outcomes(args.output / f"{a}.xml") for a in ("previous", "candidate")
    )
    if any(v["state"] == "error" for arm in (before, after) for v in arm.values()):
        raise RuntimeError(
            "Setup errors invalidate the comparison; inspect preserved logs"
        )
    expected = json.loads(
        (args.prior / "regression_summary.json").read_text(encoding="utf-8")
    )["global"]["failure_ids"]
    matrix = []
    for node in expected:
        name = node.split("::", 1)[1]
        b, a = before[name], after[name]
        same_cause = all(
            "Clustering membership provenance is incomplete" in r["trace"]
            for r in (b, a)
        )
        matrix.append(
            {
                "test_id": node,
                "previous": b["state"],
                "candidate": a["state"],
                "classification": "preexisting_contract_producer_mismatch"
                if b["state"] == a["state"] == "failed" and same_cause
                else "requires_investigation",
                "cause": "Full artifact producer sets membership_consistency_verified=False; pre-existing local validator requires True",
                "cause_location": "src/flir_pipeline/explorer/organization.py:397; organization_contract.py:251",
                "evidence": f"previous.xml / candidate.xml / {name}",
                "general_contract_effect": "No observed failure in general contracts; this failure blocks the separate organization export",
                "target_assertion": "Export setup fails before intended downstream assertion"
                if any(
                    s in name
                    for s in (
                        "integrity_failures",
                        "contract_rejects",
                        "duplicate_primary",
                        "io_failure",
                        "concurrent_source",
                    )
                )
                else "Organization export rejected by provenance validation",
                "action": "Preserve existing code; no correction authorized for pre-existing mismatch",
            }
        )
    save(args.output / "failure_matrix.json", matrix)
    with (args.output / "failure_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    differences = [
        name
        for name in after
        if before.get(name, {}).get("state") != after[name]["state"]
    ]
    save(
        args.output / "comparison.json",
        {
            "same_test_ids": set(before) == set(after),
            "state_differences": differences,
            "previous_passed": sum(v["state"] == "passed" for v in before.values()),
            "candidate_passed": sum(v["state"] == "passed" for v in after.values()),
            "previous_failed": sum(v["state"] == "failed" for v in before.values()),
            "candidate_failed": sum(v["state"] == "failed" for v in after.values()),
            "all_34_classified_preexisting": all(
                r["classification"].startswith("preexisting") for r in matrix
            ),
            "scope": "Pre-general integration control, preserving prior local organization changes; not pristine HEAD",
            "global_ci_green": False,
        },
    )


if __name__ == "__main__":
    main()
