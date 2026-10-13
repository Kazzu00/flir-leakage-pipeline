"""Strict adapter for labeled audits; no fabricated historical or temporal data."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.data.adendo import VERSION as AUDIT_VERSION
from flir_pipeline.data.adendo import _json
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.manifest import _stable_id
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.similarity.storage import file_sha256, read_json
from flir_pipeline.splitting.base import SPLITS, SplitConfig, targets_from_manifest
from flir_pipeline.splitting.construction import make_groups, record_statistics

VERSION = "labeled_candidates_v1"
STATES = {
    "exact_bytes",
    "confirmed_provenance",
    "visual_candidate",
    "unresolved",
    "rejected",
}
AUDIT_FILES = {
    "candidate_manifest.json",
    "candidate_manifest.parquet",
    "annotation_audit.json",
    "relations.json",
    "README.md",
}


@dataclass(frozen=True)
class LabeledConfig:
    """S3 constraints are an explicit sensitivity hypothesis, never adjudication.

    Ratio inference is restricted to historical occurrences. Balance targets are
    soft objectives; an optional deviation ceiling is a publication gate, not a
    claim that the MILP optimizes under an additional hard constraint.
    """

    balance: SplitConfig = field(default_factory=SplitConfig)
    sensitivity_states: tuple[str, ...] = (
        "visual_candidate",
        "unresolved",
        "confirmed_provenance",
    )
    review_conflict_policy: str = "error"
    max_relative_record_deviation: float | None = None
    protocol_version: str = VERSION

    def __post_init__(self):
        if isinstance(self.balance, dict):
            object.__setattr__(self, "balance", SplitConfig(**self.balance))
        object.__setattr__(
            self, "sensitivity_states", tuple(sorted(self.sensitivity_states))
        )
        if self.balance.strategy != "cluster_aware" or self.protocol_version != VERSION:
            raise ValueError(
                "Labeled adapter requires its own protocol and the MILP balance configuration"
            )
        if (
            not self.sensitivity_states
            or len(set(self.sensitivity_states)) != len(self.sensitivity_states)
            or not set(self.sensitivity_states) <= STATES - {"exact_bytes", "rejected"}
        ):
            raise ValueError(
                "Sensitivity must explicitly select positive evidence states"
            )
        if self.review_conflict_policy not in {"error", "reject_pair"}:
            raise ValueError(
                "Review conflicts require error or explicit conservative reject_pair"
            )
        if self.max_relative_record_deviation is not None and (
            not np.isfinite(self.max_relative_record_deviation)
            or self.max_relative_record_deviation < 0
        ):
            raise ValueError("Invalid balance publication ceiling")


@dataclass
class LabeledSource:
    manifest: pd.DataFrame
    statistics: pd.DataFrame
    relations: list[dict]
    summary: dict
    signatures: dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_name(name: str) -> bool:
    return (
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and not any(c in name for c in "/\\:")
    )


def load_labeled_audit(
    directory: Path,
    *,
    expected_dataset_id: str | None = None,
    expected_audit_id: str | None = None,
) -> LabeledSource:
    """Verify receipts and reconstruct every matched annotation from original bytes.

    This validates the audit snapshot, not image decoding or the original ZIPs.
    `verify_original_sources` separately verifies those archive byte checksums.
    Orphans are validated and retained upstream but never become fake images.
    """
    summary = read_json(directory / "summary.json")
    _require(summary["schema_version"] == AUDIT_VERSION, "Unsupported audit schema")
    _require(
        set(summary["artifacts"]) == AUDIT_FILES, "Incomplete audit artifact receipt"
    )
    for name, digest in summary["artifacts"].items():
        _require(
            file_sha256(directory / name) == digest, f"Audit checksum mismatch: {name}"
        )
    _require(
        hashlib.sha256(_json(summary["configuration"]).encode()).hexdigest()
        == summary["audit_id"],
        "Audit identity mismatch",
    )
    for key, expected in (
        ("dataset_id", expected_dataset_id),
        ("audit_id", expected_audit_id),
    ):
        _require(expected is None or summary[key] == expected, f"Unexpected {key}")
    envelopes = {
        name: read_json(directory / name)
        for name in (
            "candidate_manifest.json",
            "annotation_audit.json",
            "relations.json",
        )
    }
    for envelope in envelopes.values():
        _require(
            envelope["schema_version"] == AUDIT_VERSION
            and envelope["dataset_id"] == summary["dataset_id"],
            "Audit envelope identity mismatch",
        )
    images = envelopes["candidate_manifest.json"]["records"]
    manifest = pd.DataFrame(images).sort_values("frame_id").reset_index(drop=True)
    parquet = (
        pd.read_parquet(directory / "candidate_manifest.parquet")
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    # JSON round trip normalizes nullable parquet arrays without losing source data.
    _require(
        json.loads(parquet.to_json(orient="records"))
        == json.loads(manifest.to_json(orient="records")),
        "Manifest JSON/parquet disagreement",
    )
    _require(
        dataset_id_from_manifest(manifest) == summary["dataset_id"],
        "Dataset identity mismatch",
    )
    _require(
        set(manifest.cohort) == {"historical", "adendo"},
        "Expected both labeled cohorts",
    )
    _require(
        manifest.loc[manifest.cohort == "historical", "original_split"]
        .isin(SPLITS)
        .all(),
        "Historical membership incomplete",
    )
    _require(
        manifest.loc[manifest.cohort == "adendo", "original_split"].isna().all(),
        "Adendo must retain original_split=null",
    )
    labels = envelopes["annotation_audit.json"]["records"]
    label_index = {row["label_id"]: row for row in labels}
    _require(len(label_index) == len(labels), "Duplicate label identity")
    frame_index = {row["frame_id"]: row for row in images}
    parsed = {}
    for label in labels:
        raw = base64.b64decode(label["original_bytes_base64"], validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        result = validate_label_bytes(raw)
        _require(
            digest == label["label_sha256"]
            and label["label_id"]
            == _stable_id(label["source_archive"], label["source_member_path"], digest),
            "Label byte identity mismatch",
        )
        _require(
            result.label_valid and set(result.classes_present) <= set(range(5)),
            "Invalid label or unknown class",
        )
        _require(
            json.loads(json.dumps(asdict(result))) == label["validation"],
            "Annotation validation disagrees with original bytes",
        )
        _require(
            dict(Counter(str(box[0]) for box in result.boxes)) == label["class_counts"]
            and not label["unknown_class_ids"],
            "Class counts disagree with original bytes",
        )
        _require(
            label["pairing_status"] in {"matched", "orphan"}, "Ambiguous label pairing"
        )
        _require(
            (label["frame_ids"] == [])
            if label["pairing_status"] == "orphan"
            else len(label["frame_ids"]) == 1 and label["frame_ids"][0] in frame_index,
            "Invalid label coverage",
        )
        parsed[label["label_id"]] = result
    instances, used = [], set()
    for row in images:
        _require(
            row["content_id"] == row["image_sha256"]
            and row["frame_id"]
            == _stable_id(
                row["source_archive"], row["source_member_path"], row["content_id"]
            ),
            "Occurrence/content identity inconsistent",
        )
        _require(
            row["manifest_version"] == AUDIT_VERSION
            and row["pairing_status"] == "matched"
            and row["label_valid"] is True
            and row["image_decode_valid"] is True
            and len(row["label_ids"]) == 1,
            "Invalid or incomplete occurrence annotation",
        )
        key = row["label_ids"][0]
        _require(
            key in label_index and key not in used,
            "Missing or reused occurrence annotation",
        )
        label, result = label_index[key], parsed[key]
        _require(
            label["frame_ids"] == [row["frame_id"]]
            and label["pairing_status"] == "matched"
            and label["cohort"] == row["cohort"]
            and label["original_split"] == row["original_split"]
            and label["basename"] == row["basename"]
            and label["label_sha256"] == row["label_sha256"]
            and result.label_empty == row["label_empty"],
            "Annotation lineage mismatch",
        )
        used.add(key)
        instances.extend(
            {"frame_id": row["frame_id"], "class_id": box[0]} for box in result.boxes
        )
    _require(
        used == {r["label_id"] for r in labels if r["pairing_status"] == "matched"},
        "Incomplete matched label coverage",
    )
    manifest["num_objects"] = [parsed[r[0]].num_objects for r in manifest.label_ids]
    statistics = record_statistics(
        manifest, pd.DataFrame(instances, columns=["frame_id", "class_id"])
    )
    relations = envelopes["relations.json"]["records"]
    _require(
        len({r["relation_id"] for r in relations}) == len(relations),
        "Duplicate relation identity",
    )
    exact_pairs = set()
    conflicts = 0
    for edge in relations:
        _require(
            edge["state"] in STATES and edge["split_constraint_applied"] is False,
            "Invalid source relation state",
        )
        a, b = edge["left_frame_id"], edge["right_frame_id"]
        _require(
            a in frame_index and b in frame_index and a != b,
            "Unknown relation endpoint",
        )
        left, right = frame_index[a], frame_index[b]
        _require(
            edge["left_content_id"] == left["content_id"]
            and edge["right_content_id"] == right["content_id"],
            "Relation content lineage mismatch",
        )
        _require(
            edge["relation_id"]
            == _stable_id(a, b, edge["state"], _json(edge["evidence"])),
            "Relation identity mismatch",
        )
        if edge["state"] == "exact_bytes":
            _require(
                left["content_id"] == right["content_id"]
                and edge["review_status"] == "confirmed_by_bytes",
                "False exact relation",
            )
            exact_pairs.add(tuple(sorted((a, b))))
            conflict = (
                parsed[left["label_ids"][0]].canonical
                != parsed[right["label_ids"][0]].canonical
            )
            _require(
                edge["annotation_difference"]["annotation_conflict"] == conflict,
                "Incorrect annotation conflict",
            )
            conflicts += int(conflict)
        elif edge["state"] in {"confirmed_provenance", "rejected"}:
            _require(
                edge["review_status"] == "reviewed"
                and all(
                    isinstance(edge["evidence"].get(k), str)
                    and edge["evidence"][k].strip()
                    for k in ("reviewer", "reviewed_at", "additional_evidence")
                ),
                "Review evidence incomplete",
            )
        else:
            _require(
                edge["review_status"] == "pending",
                "Provisional review state inconsistent",
            )
    expected_pairs = {
        tuple(sorted(pair))
        for _, group in manifest.groupby("content_id")
        for pair in combinations(group.frame_id, 2)
    }
    _require(
        exact_pairs == expected_pairs
        and sum(e["state"] == "exact_bytes" for e in relations) == len(expected_pairs),
        "Exact relation coverage incomplete",
    )
    sizes = manifest.groupby("content_id").size()
    candidates = [
        r for r in relations if r["state"] in {"visual_candidate", "unresolved"}
    ]
    observations = {
        "total_occurrences": len(manifest),
        "unique_contents": len(sizes),
        "exact_duplicate_groups": int((sizes > 1).sum()),
        "exact_duplicate_occurrences": int(sizes[sizes > 1].sum()),
        "exact_duplicate_additional_occurrences": len(manifest) - len(sizes),
        "exact_duplicate_relation_pairs": len(exact_pairs),
        "exact_annotation_conflict_pairs": conflicts,
        "candidate_unique_pairs": len(
            {(r["left_frame_id"], r["right_frame_id"]) for r in candidates}
        ),
        "candidate_evidence_records": len(candidates),
        "relation_states": dict(Counter(r["state"] for r in relations)),
    }
    for key, value in observations.items():
        _require(summary[key] == value, f"Audit count mismatch: {key}")
    for cohort in ("historical", "adendo"):
        rows = manifest.loc[manifest.cohort == cohort]
        stats = statistics.loc[statistics.frame_id.isin(rows.frame_id)]
        cohort_labels = [r for r in labels if r["cohort"] == cohort]
        expected = {
            "images": len(rows),
            "unique_contents": rows.content_id.nunique(),
            "labels": len(cohort_labels),
            "orphan_labels": sum(
                r["pairing_status"] == "orphan" for r in cohort_labels
            ),
            "matched_objects": int(stats[[f"class_{k}" for k in range(5)]].sum().sum()),
            "empty_matched_labels": int(stats.empty_annotation_count.sum()),
            "original_splits": dict(Counter(rows.original_split.dropna())),
            "matched_class_counts": {
                str(k): int(stats[f"class_{k}"].sum())
                for k in range(5)
                if stats[f"class_{k}"].sum()
            },
        }
        for key, value in expected.items():
            _require(
                summary["counts"][cohort][key] == value,
                f"Cohort count mismatch: {cohort}/{key}",
            )
    signatures = {
        **summary["artifacts"],
        "summary.json": file_sha256(directory / "summary.json"),
    }
    return LabeledSource(manifest, statistics, relations, summary, signatures)


def verify_original_sources(
    source: LabeledSource, root: Path, evidence: Path | None
) -> dict:
    """Stream hashes without extracting, modifying or decoding source archives."""
    config = source.summary["configuration"]
    for item in config["sources"].values():
        _require(_safe_name(item["name"]), "Unsafe source archive name")
        _require(
            file_sha256(root / item["name"]) == item["sha256"],
            "Original ZIP checksum mismatch",
        )
    if config["evidence_sha256"] is not None:
        _require(
            evidence is not None and file_sha256(evidence) == config["evidence_sha256"],
            "Original evidence checksum mismatch or missing evidence",
        )
    return {
        "archive_sha256_verified": config["sources"],
        "evidence_sha256_verified": config["evidence_sha256"],
    }


def population(source: LabeledSource, name: str, config: LabeledConfig):
    _require(name in {"historical", "expanded"}, "Unknown population")
    historical = source.manifest.loc[source.manifest.cohort == "historical"]
    ratios = targets_from_manifest(historical, config.balance)
    manifest = (
        (historical if name == "historical" else source.manifest)
        .copy()
        .reset_index(drop=True)
    )
    stats = source.statistics.loc[
        source.statistics.frame_id.isin(manifest.frame_id)
    ].reset_index(drop=True)
    return manifest, stats, ratios


def sensitivity_groups(
    manifest: pd.DataFrame, relations: list[dict], policy: str, config: LabeledConfig
):
    """Connected components over selected content edges in the induced population.

    A rejected pair vetoes all its positive evidence. Confirmed/rejected review
    conflicts fail by default, even for S1/S2. Rejection is not a cannot-link
    constraint; an S3 transitive path to a rejected pair is conservatively refused.
    Integer labels are only an internal bridge to the existing group verifier,
    and do not claim that an embedding clustering was performed.
    """
    _require(policy in {"S1", "S2", "S3"}, "Unknown labeled policy")
    ids = sorted(manifest.content_id.unique())
    frame_ids = set(manifest.frame_id)
    edges = [
        e
        for e in relations
        if e["left_frame_id"] in frame_ids and e["right_frame_id"] in frame_ids
    ]
    by_pair = defaultdict(list)
    for edge in edges:
        by_pair[
            tuple(sorted((edge["left_content_id"], edge["right_content_id"])))
        ].append(edge)
    parent = {cid: cid for cid in ids}

    def find(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    applied, rejected, conflicts = [], [], []
    for pair, evidence in sorted(by_pair.items()):
        states = {e["state"] for e in evidence}
        if "rejected" in states:
            _require(
                pair[0] != pair[1],
                "Rejected relation contradicts mandatory exact identity",
            )
            rejected.append(pair)
            if "confirmed_provenance" in states:
                conflicts.append(pair)
                _require(
                    config.review_conflict_policy == "reject_pair",
                    "Conflicting reviews require an explicit resolution policy",
                )
            continue
        if policy == "S3":
            selected = [e for e in evidence if e["state"] in config.sensitivity_states]
            if selected:
                a, b = find(pair[0]), find(pair[1])
                parent[max(a, b)] = min(a, b)
                applied.extend(e["relation_id"] for e in selected)
    if policy == "S3":
        _require(
            all(find(a) != find(b) for a, b in rejected),
            "Transitive sensitivity component contradicts rejected evidence",
        )
    components = defaultdict(list)
    for cid in ids:
        components[find(cid)].append(cid)
    joined = sorted(tuple(v) for v in components.values() if len(v) > 1)
    labels = pd.Series(-1, index=ids, dtype="int64") if policy == "S3" else None
    if labels is not None:
        for i, members in enumerate(joined):
            labels.loc[list(members)] = i
    groups = make_groups(ids, labels)
    selection = {
        "available_evidence_records": len(edges),
        "excluded_outside_population_records": len(relations) - len(edges),
        "imposed_relation_ids": sorted(applied),
        "imposed_as_hypothetical_constraints": policy == "S3",
        "review_conflict_policy": config.review_conflict_policy,
        "rejected_content_pairs": [list(p) for p in rejected],
        "conflicting_review_pairs": [list(p) for p in conflicts],
        "component_labels_are_clustering_results": False,
    }
    return groups, labels, selection
