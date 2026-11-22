"""GPU extraction followed by candidate splitting; no protected-test anchoring."""

import os
import sys
from pathlib import Path

from flir_pipeline.closeout.features import extract
from flir_pipeline.closeout.io import write_json
from flir_pipeline.closeout.splits import build


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run under Slurm GPU allocation")
    dataset, work, config = map(Path, sys.argv[1:])
    work.mkdir(parents=True, exist_ok=True)
    try:
        stores = extract(dataset, work / "features", "cuda", 8)
        report = build(
            dataset,
            config,
            work / "Dataset Con Splits",
            {name: Path(v["path"]) for name, v in stores.items()},
        )
    except Exception as exc:
        # Append-only operation evidence; feature checkpoints remain resumable.
        import uuid

        write_json(
            work / f"BLOCKED-{uuid.uuid4().hex}.json",
            {
                "state": "BLOCKED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "split_created": False,
            },
        )
        raise
    print(report["state"])


if __name__ == "__main__":
    main()
