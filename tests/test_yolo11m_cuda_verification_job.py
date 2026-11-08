"""Execute the job's orchestration with offline stand-ins, never Slurm/CUDA."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "scripts/hypatia/yolo11m_cuda_verify.sbatch"


def bash_binary():
    if os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).resolve().parents[1] / "bin/bash.exe" if git else None
        if candidate and candidate.is_file():
            return str(candidate)
    else:
        candidate = shutil.which("bash")
        if candidate:
            return candidate
    pytest.skip("Bash is needed for offline shell orchestration tests")


BOOTSTRAP = r'''
git() { printf '%s\n' fixture_commit; }
scontrol() { printf '%s\n' fixture_slurm; }
module() { printf '%s\n' fixture_modules; }
fake_python() {
    printf '%s\n' "$*" >> "$FAKE_CALLS"
    case "$*" in
      *'--case loss'*) return "$FAKE_LOSS_RC" ;;
      *'--case smoke'*) return "$FAKE_SMOKE_RC" ;;
      *) return 99 ;;
    esac
}
source "$1"
'''


@pytest.mark.parametrize("loss_rc,smoke_rc", [(0, 0), (7, 0), (0, 9)])
def test_one_smoke_only_after_paired_microstep(tmp_path, loss_rc, smoke_rc):
    calls = tmp_path / "calls.txt"
    output = tmp_path / "new-attempt"
    environment = {
        **os.environ,
        "SLURM_JOB_ID": "synthetic-fixture", "SLURM_CPUS_PER_TASK": "2",
        "FLIR_REPO": ROOT.as_posix(), "FLIR_PYTHON": "fake_python",
        "FLIR_DIAGNOSTIC_SCRIPT": "unchanged-diagnostic.py",
        "FLIR_CUDA_OUTPUT": output.as_posix(), "FAKE_CALLS": calls.as_posix(),
        "FAKE_LOSS_RC": str(loss_rc), "FAKE_SMOKE_RC": str(smoke_rc),
    }
    command = [bash_binary(), "-c", BOOTSTRAP, "fixture", JOB.as_posix()]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=20, check=False)
    observed = calls.read_text().splitlines()
    assert "--case loss --size 640 --profile configured" in observed[0]
    assert len(observed) == (1 if loss_rc else 2)
    if loss_rc:
        assert result.returncode != 0
        assert "SKIPPED" in (output / "smoke-status.txt").read_text()
    else:
        assert "--case smoke --size 640 --profile configured" in observed[1]
        assert result.returncode == smoke_rc, result.stderr
    # An existing output cannot be reused, even after a failed case.
    before = (output / "exit-codes.txt").read_bytes()
    repeated = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=20, check=False)
    assert repeated.returncode != 0
    assert calls.read_text().splitlines() == observed
    assert (output / "exit-codes.txt").read_bytes() == before
