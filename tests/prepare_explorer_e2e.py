"""Prepare the existing synthetic organization fixture for offline browser tests.

This helper is test support, not a scientific exporter entry point. It creates
only colored synthetic images and stored synthetic memberships; no model fits,
FAC files, production runtime directories, or frozen results are consumed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from test_organization_export import evidence

from flir_pipeline.explorer.organization import export_organization, timelines
from flir_pipeline.explorer.organization_contract import OrganizationExport


def candidate_scenario(root):
    """A separate coherent synthetic view, never evidence or a scientific export.

    Keep the unmodified producer output for cross-language parity. The UX test
    also needs a stored candidate, whose synthetic endpoints/receipts are bound
    here and checked by the complete relational contract before transport.
    """
    data = {
        key: json.loads((root / "export" / f"{key}.json").read_text(encoding="utf-8"))
        for key in OrganizationExport.model_fields
    }
    contents = data["contents"]
    video_content = contents[1]
    original = next(
        row
        for row in data["records"]
        if row["record_id"] == video_content["representative_record_id"]
    )
    appearance = {
        **original,
        "record_id": "SYNTHETIC-video-appearance",
        "cohort": "video_evidence",
        "original_split": None,
        "class_ids": None,
        "class_names": None,
        "num_objects": None,
        "label_empty": None,
    }
    data["records"].append(appearance)
    video_content["record_ids"].append(appearance["record_id"])
    data["timelines"] = timelines(data["records"], contents)
    pair = {
        "evidence_artifact_id": "SYNTHETIC-candidate-source",
        "candidate_id": "SYNTHETIC-candidate-only",
        "labeled_content_id": contents[0]["content_id"],
        "video_content_id": video_content["content_id"],
        "clip_cosine": 0.4,
        "dinov2_cosine": 0.6,
        "clip_rank": 1,
        "dinov2_rank": 1,
        "clip_topk": True,
        "dinov2_topk": True,
        "both_topk": True,
        "mean_reciprocal_rank": 1.0,
        "video_occurrence_count": 1,
        "video_sequence_count": 1,
    }
    data["candidate_pairs"] = [pair]
    source = json.dumps(
        {"scientific_result": False, "fixture_candidate": pair}, sort_keys=True
    ).encode()
    (root / "SYNTHETIC_CANDIDATE_SOURCE.json").write_bytes(source)
    manifest = data["manifest"]
    digest = manifest["manifest_sha256"]
    manifest["evidence_sources"].append(
        {
            "artifact_id": pair["evidence_artifact_id"],
            "artifact_kind": "labeled_video_link_candidates",
            "metadata_sha256": hashlib.sha256(source).hexdigest(),
            "output_checksums": {
                "SYNTHETIC_CANDIDATE_SOURCE.json": hashlib.sha256(source).hexdigest()
            },
            "labeled_manifest_binding": {
                "historical_manifest_sha256": digest,
                "current_manifest_sha256": digest,
                "source_manifest_reserialized": False,
                "exact_tabular_identity_verified": True,
            },
        }
    )
    manifest["record_count"] = len(data["records"])
    manifest["candidate_pair_count"] = 1
    manifest["limitations"].append(
        "SYNTHETIC UX fixture only; no real similarity or dependency evidence"
    )
    # Fail on inconsistent cohorts, joins, counts, semantics or source identities.
    OrganizationExport.model_validate(data)
    destination = root / "export-with-candidates"
    destination.mkdir()
    for key, value in data.items():
        if key == "manifest":
            continue
        raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
        name = f"{key}.json"
        (destination / name).write_bytes(raw)
        manifest["file_sha256"][name] = hashlib.sha256(raw).hexdigest()
    for name, expected in manifest["file_sha256"].items():
        path = destination / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((root / "export" / name).read_bytes())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    OrganizationExport.model_validate(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.output.is_dir() or any(args.output.iterdir()):
        raise ValueError("Use an existing, empty disposable directory")
    inputs, *_ = evidence.__wrapped__(args.output)
    result = export_organization(**inputs, include_previews=True)
    assert result["labeled_record_count"] == 8
    candidate_scenario(args.output)
    (args.output / "SYNTHETIC_TEST_ONLY.json").write_text(
        json.dumps(
            {
                "purpose": "offline browser contract verification; not FAC evidence",
                "scientific_result": False,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
