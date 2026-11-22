"""Exercise actual shell argument routing, including Slurm's relocated script."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def bash_path():
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    return str(git_bash) if git_bash.exists() else shutil.which("bash")


@pytest.mark.parametrize(
    "name",
    [
        "preflight_oe4.sh",
        "submit_oe4.sh",
        "oe4.sbatch",
        "aggregate_oe4.sh",
        "denoised_closeout.sbatch",
        "preflight_oe4.sbatch",
        "denoised_train.sbatch",
        "submit_denoised.sh",
    ],
)
def test_shell_syntax(name):
    bash = bash_path()
    if not bash:
        pytest.skip("Bash unavailable")
    subprocess.run(
        [bash, "-n", f"scripts/hypatia/{name}"], check=True, capture_output=True
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_spooled_worker_uses_explicit_bundle_root(tmp_path, index):
    bash = bash_path()
    if not bash:
        pytest.skip("Bash unavailable")
    spool = tmp_path / "slurm-spool"
    spool.mkdir()
    script = spool / "slurm_script"
    shutil.copyfile("scripts/hypatia/oe4.sbatch", script)
    bundle = tmp_path / "bundle with space"
    bundle.mkdir()
    interpreter = tmp_path / "fake-python"
    interpreter.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$CALL_LOG"\n',
        encoding="utf-8",
        newline="\n",
    )
    interpreter.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": str(index),
        "SLURM_JOB_ID": "synthetic",
        "CALL_LOG": log.as_posix(),
    }
    result = subprocess.run(
        [
            bash,
            script.as_posix(),
            "train",
            "config.yaml",
            interpreter.as_posix(),
            bundle.as_posix(),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[-5:] == [
        "-m",
        "flir_pipeline.closeout",
        "oe4-train",
        "config.yaml",
        str(index),
    ]
    assert "scripts/closeout/verify_bundle.py" in calls
