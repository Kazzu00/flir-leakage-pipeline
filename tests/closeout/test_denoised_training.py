import pandas as pd
from PIL import Image

from flir_pipeline.closeout.denoised_training import materialize, source_rows
from flir_pipeline.closeout.io import read_json, seal, verify, write_json
from flir_pipeline.similarity.storage import file_sha256


def test_exploratory_excludes_entire_risky_groups_without_reassigning(tmp_path):
    candidate, anchors, publication = (
        tmp_path / name for name in ("candidate", "anchors", "publication")
    )
    rows, exceptions = [], []
    for i, split in enumerate(["train", "train", "train", "val", "val", "test"]):
        image, label = (
            candidate / "images" / split / f"{i}.png",
            candidate / "labels" / split / f"{i}.txt",
        )
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (i, 0, 0)).save(image)
        label.write_text("".join(f"{c} .5 .5 .1 .1\n" for c in range(5)))
        rows.append(
            {
                "frame_id": f"d{i}",
                "content_id": file_sha256(image),
                "image_sha256": file_sha256(image),
                "label_sha256": file_sha256(label),
                "image_path": f"images/{i}.png",
                "label_path": f"labels/{i}.txt",
                "new_split": split,
                "group_id": "risky" if i < 2 else f"g{i}",
                **{f"class_{c}": 1 for c in range(5)},
            }
        )
        exceptions.append(
            {
                "denoised_frame_id": f"d{i}",
                "status": "UNMAPPED" if i == 3 else "NOMINAL_ONLY",
                "candidate_frame_ids": "protected" if i == 0 else f"orig{i}",
                "candidate_test_ids": "protected" if i == 0 else "",
            }
        )
    pd.DataFrame(rows).to_csv(candidate / "memberships.csv", index=False)
    write_json(candidate / "audit.json", {"state": "CANDIDATE_NOT_FROZEN"})
    write_json(
        candidate / "provenance.json", {"source_receipt_sha256": "fixture-source"}
    )
    seal(candidate, {"state": "CANDIDATE_NOT_FROZEN"})
    (publication / "sources/proposal").mkdir(parents=True)
    (publication / "sources/audit").mkdir()
    pd.DataFrame([{"frame_id": "protected", "content_id": "original-test"}]).to_parquet(
        publication / "sources/audit/candidate_manifest.parquet"
    )
    pd.DataFrame([{"frame_id": "protected", "content_id": "original-test"}]).to_parquet(
        publication / "sources/proposal/test_records.parquet"
    )
    pd.DataFrame(
        [{"group_id": "protected-group", "content_id": "original-test"}]
    ).to_parquet(publication / "sources/proposal/protection_groups.parquet")
    seal(publication, {"state": "synthetic publication"})
    anchors.mkdir()
    pd.DataFrame(exceptions).to_csv(
        anchors / "correspondence_exceptions.csv", index=False
    )
    write_json(
        anchors / "audit.json",
        {
            "verified_anchors": 0,
            "source_dataset_receipt_sha256": "fixture-source",
            "publication_receipt_sha256": file_sha256(publication / "receipt.json"),
            "protected_records_sha256": file_sha256(
                publication / "sources/proposal/test_records.parquet"
            ),
        },
    )
    seal(anchors, {"state": "BLOCKED_PROTECTED_FREEZE"})
    output = tmp_path / "exploratory"
    result = materialize(candidate, anchors, publication, output)
    assert result["excluded_records"] == 2
    assert result["counts"] == {"val": 2, "train": 1, "test": 1}
    assert result["unresolved_retained"] == 1
    assert result["common_test_evaluation_forbidden"] is True
    assert result["confirmatory_denoising_effect"] is False
    assert len(source_rows(output)) == 3  # Internal test is a catalogue only.
    assert len(list((output / "images").glob("*.png"))) == 3
    assert set(pd.read_csv(output / "excluded.csv").group_id) == {"risky"}
    assert read_json(output / "audit.json")["original_split_reassigned"] is False
    verify(output)
