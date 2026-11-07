"""Reproducible external JP artifacts; no mutation of scientific source stores."""

import argparse
import importlib.metadata
import subprocess
from pathlib import Path

from flir_pipeline.explorer.paired_inputs import (
    load_inputs,
    read_json,
    safe_relative,
    write_json,
)
from flir_pipeline.utils.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["timeline", "split", "verify"])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg, pairs, receipts, reconciliation = load_inputs(args.evidence_root, args.config)
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[3]
    forbidden = [
        repo,
        safe_relative(args.evidence_root, cfg["data"]),
        safe_relative(args.evidence_root, cfg["inventory"]),
        safe_relative(args.evidence_root, cfg["hypatia"]),
        safe_relative(args.evidence_root, cfg["assessment"]),
    ]
    if any(
        output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
        for p in forbidden
    ):
        raise ValueError(
            "Output must be separate from repositories and source artifacts"
        )
    if args.stage == "timeline":
        output.mkdir(parents=True, exist_ok=False)
        from flir_pipeline.explorer.paired_timeline import build_timeline

        build_timeline(
            pairs,
            safe_relative(args.evidence_root, cfg["data"]),
            output,
            cfg,
            reconciliation,
        )
        source_files = [
            Path(__file__),
            Path(__file__).with_name("paired_inputs.py"),
            Path(__file__).with_name("paired_timeline.py"),
            repo / "src/flir_pipeline/splitting/paired.py",
            repo / "src/flir_pipeline/splitting/construction.py",
            args.config,
        ]
        provenance = {
            "inputs_sha256": receipts,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "code_sha256": {
                str(p.relative_to(repo))
                if p.is_relative_to(repo)
                else p.name: sha256_file(p)
                for p in source_files
                if p.exists()
            },
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "pandas", "pillow", "pyyaml"]
            },
            "configuration": cfg,
            "training_allowed": False,
        }
        write_json(output / "provenance.json", provenance)
        write_json(
            output / "timeline_receipt.json",
            {
                name: sha256_file(output / name)
                for name in [
                    "timeline_records.json",
                    "timeline_manifest.csv",
                    "timeline_comparison.html",
                    "timeline_verification.json",
                    "provenance.json",
                ]
            },
        )
    else:
        for name, checksum in read_json(output / "timeline_receipt.json").items():
            if sha256_file(output / name) != checksum:
                raise ValueError(f"Timeline changed: {name}")
        frozen_inputs = read_json(output / "provenance.json")["inputs_sha256"]
        additional = set(receipts) - set(frozen_inputs)
        if any(receipts.get(k) != v for k, v in frozen_inputs.items()) or any(
            not k.endswith(".tar") for k in additional
        ):
            raise ValueError("Inputs differ from timeline; create a new delivery")
        # A later verifier may strengthen a prior receipt with full TAR hashes;
        # existing bound inputs must still match exactly. Preserve the old receipt.
        write_json(
            output / "input_revalidation.json",
            {
                "inputs_sha256": receipts,
                "additional_archive_bindings": sorted(additional),
                "all_prior_input_hashes_match": True,
            },
        )
        from flir_pipeline.splitting.paired import build_splits, verify_saved

        if args.stage == "split":
            build_splits(
                read_json(output / "timeline_records.json"),
                cfg,
                args.evidence_root,
                output,
            )
        else:
            print(verify_saved(output))
    print(f"{args.stage}: {output}")


if __name__ == "__main__":
    main()
