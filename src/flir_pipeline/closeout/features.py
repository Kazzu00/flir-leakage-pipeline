"""Denoised adapter to the existing content-level raw/L2/resumable feature store."""

from pathlib import Path

from flir_pipeline.closeout.io import read_json, write_json
from flir_pipeline.closeout.splits import load_manifest
from flir_pipeline.features.storage import (
    extract_to_store,
    verify_features_against_manifest,
)


def extract(dataset: Path, output: Path, device="cpu", batch_size=8):
    rows = load_manifest(dataset)
    if output.resolve().is_relative_to(
        dataset.resolve()
    ) or dataset.resolve().is_relative_to(output.resolve()):
        raise ValueError("Features must be outside source dataset")
    output.mkdir(parents=True, exist_ok=True)
    # Only image identity/locators and identity hashes enter extraction.
    manifest = rows[
        [
            "frame_id",
            "content_id",
            "image_sha256",
            "label_sha256",
            "manifest_version",
            "image_path",
        ]
    ]
    manifest_path = output / "manifest.parquet"
    if manifest_path.exists():
        import pandas as pd

        pd.testing.assert_frame_equal(pd.read_parquet(manifest_path), manifest)
    else:
        manifest.to_parquet(manifest_path, index=False)
    from flir_pipeline.features.clip import CLIPExtractor
    from flir_pipeline.features.dinov2 import DinoV2Extractor

    result = {}
    for name, factory in (("dinov2", DinoV2Extractor), ("clip", CLIPExtractor)):
        directory = extract_to_store(
            manifest_path,
            output_root=output / "stores",
            images_root=dataset,
            batch_size=batch_size,
            seed=0,
            extractor_factory=lambda factory=factory: factory(
                device=device,
                batch_size=batch_size,
                local_files_only=True,
                require_resolved_revision=True,
            ),
        )
        check = verify_features_against_manifest(directory, manifest)
        if not check["reproducible_full_dataset_valid"]:
            raise ValueError("Full feature store failed provenance/coverage checks")
        result[name] = {
            "path": str(directory.resolve()),
            "feature_space_id": read_json(directory / "metadata.json")[
                "feature_space_id"
            ],
            "verification": check,
        }
    write_json(output / "stores.json", result)
    return result
