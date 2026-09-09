"""Content-level image quality diagnostics, independent from learned features."""

from __future__ import annotations

import json
import platform
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from tqdm import tqdm

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.preprocessing import decode_zip_image


def _entropy(gray: np.ndarray) -> float:
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _laplacian_variance(gray: np.ndarray) -> float:
    values = gray.astype(np.float32)
    padded = np.pad(values, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def _phash(image: Image.Image) -> str:
    resized = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = dctn(resized, type=2, norm="ortho")[:8, :8]
    threshold = np.median(coefficients[1:, 1:])
    bits = (coefficients > threshold).astype(np.uint8).flatten()
    return format(int("".join(str(bit) for bit in bits), 2), "016x")


def _dhash(image: Image.Image) -> str:
    resized = np.asarray(image.convert("L").resize((9, 8)), dtype=np.uint8)
    bits = (resized[:, 1:] > resized[:, :-1]).astype(np.uint8).flatten()
    return format(int("".join(str(bit) for bit in bits), 2), "016x")


def run_diagnostics(
    manifest_path: Path,
    images_archive: Path,
    output_root: Path = Path("artifacts/features/diagnostics"),
    dataset_id: str | None = None,
    limit_content: int | None = None,
    seed: int = 0,
) -> Path:
    """Write content-level QA/EDA measurements and metadata from read-only images.

    Pixel statistics assume 8-bit images; entropy and edge variance use grayscale.
    pHash/dHash describe appearance and are not exact-content identities. These
    diagnostics neither filter images nor concatenate with DINOv2/CLIP vectors.
    Return the local dataset directory; sampled runs should use a separate root.
    """
    manifest = pd.read_parquet(manifest_path)
    dataset_id = dataset_id or dataset_id_from_manifest(manifest)
    unique = manifest.sort_values("content_id").drop_duplicates("content_id", keep="first")
    if limit_content is not None:
        if limit_content < 1:
            raise ValueError("limit_content must be positive")
        if len(unique) > limit_content:
            generator = np.random.default_rng(seed)
            indices = generator.choice(len(unique), limit_content, replace=False)
            unique = unique.iloc[np.sort(indices)]
    output = output_root / dataset_id
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(images_archive) as archive:
        for row in tqdm(
            unique.itertuples(index=False),
            total=len(unique),
            desc="Diagnostics",
            unit="image",
        ):
            image = decode_zip_image(archive, row.source_member_path)
            array = np.asarray(image, dtype=np.float32)
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
            rows.append(
                {
                    "content_id": row.content_id,
                    "representative_frame_id": row.frame_id,
                    "width": image.width,
                    "height": image.height,
                    "aspect_ratio": image.width / image.height,
                    "original_mode": image.mode,
                    "pixel_mean": float(array.mean() / 255),
                    "pixel_std": float(array.std() / 255),
                    "pixel_min": float(array.min() / 255),
                    "pixel_max": float(array.max() / 255),
                    "entropy": _entropy(gray),
                    "laplacian_variance": _laplacian_variance(gray),
                    "phash": _phash(image),
                    "dhash": _dhash(image),
                }
            )
    dataframe = pd.DataFrame(rows)
    dataframe.to_parquet(output / "diagnostics.parquet", index=False)
    metadata = {
        "dataset_id": dataset_id,
        "feature_type": "diagnostics",
        "total_records": len(manifest),
        "unique_content_ids": int(manifest["content_id"].nunique()),
        "selected_content_ids": len(dataframe),
        "seed": seed,
        "python_version": platform.python_version(),
        "created_at": datetime.now(UTC).isoformat(),
        "source_archive": images_archive.name,
        "uses_original_split": False,
        "uses_labels": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output
