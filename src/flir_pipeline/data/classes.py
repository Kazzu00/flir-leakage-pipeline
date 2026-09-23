"""Detection vocabulary with explicit evidence boundaries; background is not a box.

The original dataset YAML calls ID 4 ``SDZI``. Its canonical name follows the
publication's class order confirmed by the project owner, not an expansion of
that internal term. See docs/analysis/dataset_classes.md for evidence and limitations.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import yaml


@dataclass(frozen=True)
class DetectionClass:
    source_name: str
    class_name: str
    academic_name: str
    academic_mapping_validated: bool


DETECTION_CLASSES = MappingProxyType({
    0: DetectionClass("vehicle", "Vehicles", "Vehicles", True),
    1: DetectionClass("building", "Buildings", "Buildings", True),
    2: DetectionClass("road", "Roads", "Roads", True),
    3: DetectionClass("river", "Rivers", "Rivers", True),
    4: DetectionClass("SDZI", "Heavy Machinery", "Heavy Machinery", True),
})


def class_name(class_id: int) -> str:
    """Resolve a known detection ID without inventing an unknown/background class."""
    try:
        return DETECTION_CLASSES[class_id].class_name
    except KeyError as error:
        raise ValueError(f"Unrecognized detection class_id: {class_id}") from error


def class_display_name(class_id: int) -> str:
    return f"{class_name(class_id)} ({class_id})"


def class_catalog_rows() -> list[dict]:
    return [{"class_id": key, **asdict(value)} for key, value in DETECTION_CLASSES.items()]


def verify_class_config(archive_path: Path, member: str = "dataset_split_completo/dataset.yaml") -> dict:
    """Read only the source YAML and validate names at their actual IDs, not an offset.

    Record only portable vocabulary evidence, excluding the YAML's private path.
    The academic correspondence also uses the class order confirmed by the user;
    source-config matching alone cannot establish the meaning of SDZI.
    """
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist().count(member) != 1:
            raise ValueError("Class config member is missing or ambiguous")
        content = archive.read(member)
    config = yaml.safe_load(content)
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    expected = {key: value.source_name for key, value in DETECTION_CLASSES.items()}
    if config.get("nc") != len(expected) or names != expected:
        raise ValueError("Source class config differs from the documented detection vocabulary")
    return {
        "archive": archive_path.name,
        "member": member,
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_config_validated": True,
        "academic_mapping_validated": all(c.academic_mapping_validated for c in DETECTION_CLASSES.values()),
        "academic_mapping_basis": "original YAML IDs + publication class order confirmed by project owner on 2026-09-13; not a linguistic expansion of SDZI",
        "classes": class_catalog_rows(),
    }
