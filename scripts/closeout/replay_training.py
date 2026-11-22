"""Explicit real-data CPU replay of the count-free manifest contract, no training.

Uses the existing local historical-S1-0 publication, not claimed to be the remote
OE4 assignments/args. The remote short preflight exercises all three actual ones.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from flir_pipeline.closeout.common_test import development_records
from flir_pipeline.closeout.io import write_json
from flir_pipeline.closeout.oe4 import REQUIRED, aug_off
from flir_pipeline.closeout.oe4_runtime import execute_prepared
from flir_pipeline.closeout.training import check_cpu, prepare
from flir_pipeline.detection.annotation_review import source_bytes
from flir_pipeline.detection.scientific_executor import write_view
from flir_pipeline.similarity.storage import file_sha256


def replay(publication, source, weights, output, denoised=None):
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    output.mkdir(parents=True, exist_ok=False)
    candidate = pd.read_parquet(
        publication / "sources/audit/candidate_manifest.parquet"
    )
    memberships = pd.read_parquet(
        publication / "runs/historical-S1-0/occurrence_memberships.parquet"
    )
    rows = memberships.loc[memberships.new_split.isin(["train", "val"])].copy()
    annotation = json.loads(
        (publication / "sources/audit/annotation_audit.json").read_text()
    )["records"]
    label_lookup = {r["label_id"]: r for r in annotation}
    images, labels = [], []
    for row in rows.to_dict("records"):
        if len(row["label_ids"]) != 1:
            raise ValueError("Ambiguous original label")
        image = (
            output
            / "source/images"
            / (row["frame_id"] + Path(row["zip_member_name"]).suffix)
        )
        label = output / "source/labels" / (row["frame_id"] + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(source_bytes(source, row, row["image_sha256"]))
        label.write_bytes(
            source_bytes(source, label_lookup[row["label_ids"][0]], row["label_sha256"])
        )
        images.append(str(image.resolve()))
        labels.append(str(label.resolve()))
    rows["image_path"], rows["label_path"] = images, labels
    dataset = write_view(rows, output / "historical-native-view", strict_geometry=False)
    raw_contract = development_records(dataset, candidate)
    assert "num_objects" not in raw_contract
    raw_contract.to_parquet(output / "count_free_development.parquet", index=False)
    baseline = {
        **DEFAULT_CFG_DICT,
        **REQUIRED,
        "amp": False,
        "workers": 0,
        "cache": False,
    }
    prepared = prepare(
        raw_contract, output / "prepared-aug-off", aug_off(baseline), weights
    )
    validation = check_cpu(prepared, baseline)
    model = YOLO(str(weights))  # CPU checkpoint load only; no forward or train.
    calls = []

    class BoundaryReached(Exception):
        pass

    def stop(**settings):
        assert settings == prepared["settings"]
        calls.append(settings)
        raise BoundaryReached("Reached YOLO.train invocation without executing it")

    model.train = stop
    try:
        with patch("ultralytics.YOLO", return_value=model):
            execute_prepared(
                prepared,
                {"policy": "LOCAL_SCHEMA_REPLAY", "augmentation": "AUG_OFF"},
                lambda: None,
            )
    except BoundaryReached:
        pass
    if len(calls) != 1:
        raise ValueError("Training boundary was not reached exactly once")
    result = {
        "state": "REAL_DATA_CPU_INVOCATION_REPLAY_PASSED",
        "records": len(raw_contract),
        "source_assignment": "existing local historical-S1-0; not the remote OE4 runs",
        "source_columns": sorted(raw_contract.columns),
        "num_objects_in_input": False,
        "counts_match_native_view": prepared["counts"]
        == rows.set_index("frame_id").num_objects.to_dict(),
        "initial_weights_sha256": file_sha256(weights),
        "loader_validation": validation,
        "configuration_scope": "Pinned defaults plus known protocol controls; remote actual args verified by oe4-prepare",
        "training_invocations_intercepted": 1,
        "training_executed": False,
        "inference_executed": False,
        "gpu_used": False,
    }
    write_json(output / "replay.json", result)
    if denoised:
        from flir_pipeline.closeout.denoised_training import NAMES, source_rows

        denoised_checks = {}
        for treatment, controls in (
            ("AUG_ON", baseline),
            ("AUG_OFF", aug_off(baseline)),
        ):
            data = prepare(
                source_rows(denoised),
                output / f"denoised-{treatment}",
                controls,
                weights,
                names=NAMES,
            )
            denoised_checks[treatment] = check_cpu(data, baseline)

            def stop_denoised(_expected=data["settings"], **settings):
                assert settings == _expected
                raise BoundaryReached(
                    "Denoised model.train boundary reached without training"
                )

            model.train = stop_denoised
            try:
                with patch("ultralytics.YOLO", return_value=model):
                    execute_prepared(
                        data,
                        {
                            "policy": f"DENOISED_LOCAL_REPLAY_{treatment}",
                            "augmentation": treatment,
                            "exploratory": True,
                        },
                        lambda: None,
                    )
            except BoundaryReached:
                pass
        write_json(
            output / "denoised_replay.json",
            {
                "state": "REAL_DENOISED_CPU_BOUNDARY_REPLAY_PASSED",
                "checks": denoised_checks,
                "training_executed": False,
                "gpu_used": False,
                "common_test_evaluation_forbidden": True,
            },
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("publication", "source", "weights", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--denoised", type=Path)
    args = parser.parse_args()
    replay(args.publication, args.source, args.weights, args.output, args.denoised)
