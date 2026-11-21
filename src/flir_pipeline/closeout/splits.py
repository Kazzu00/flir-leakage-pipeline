"""Candidate denoised dependency graph and existing group-atomic MILP adapter.

The policy is explicitly new: provider frame identity + nominal scene/time +
fixed dual-encoder cosine sensitivities. It is NOT the historical C10 fit.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.closeout.io import (
    digest,
    read_json,
    seal,
    transaction,
    verify,
    write_csv,
    write_json,
)
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.storage import verify_features_against_manifest
from flir_pipeline.similarity.storage import file_sha256
from flir_pipeline.splitting.base import BALANCE_COLUMNS, CLASS_COLUMNS, SplitConfig
from flir_pipeline.splitting.construction import milp_assignment


def load_manifest(dataset: Path) -> pd.DataFrame:
    verify(dataset)
    rows = pd.read_csv(dataset / "manifest.csv", keep_default_na=False)
    rows["image_sha256"] = rows.content_id
    rows["manifest_version"] = "flir_denoised_v1"
    return rows


class Components:
    def __init__(self, ids):
        self.parent = {v: v for v in ids}

    def find(self, value):
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = sorted((self.find(a), self.find(b)))
        self.parent[b] = a


def dependency_graph(rows: pd.DataFrame, window: float, visual_edges=()):
    if not np.isfinite(window) or window < 0 or not rows.frame_id.is_unique:
        raise ValueError("Invalid temporal window or duplicate occurrence")
    graph = Components(rows.frame_id)
    edges = []

    def link(a, b, kind, evidence):
        if a == b:
            return
        graph.union(a, b)
        a, b = sorted((a, b))
        edges.append({"left": a, "right": b, "kind": kind, "evidence": evidence})

    for column, kind in [
        ("content_id", "exact_content"),
        ("source_frame_key", "provider_same_source_frame"),
    ]:
        for _, block in rows.groupby(column, sort=True):
            ids = sorted(block.frame_id)
            for other in ids[1:]:
                link(ids[0], other, kind, column)
    scenes = defaultdict(list)
    for row in rows.to_dict("records"):
        match = re.search(r"(?:^|_)scene(\d+)(?:\.|_|$)", row["nombre_original"], re.I)
        if match:
            scenes[(row["archivo_video"], row["dataset_origen"], match[1])].append(
                row["frame_id"]
            )
    for ids in scenes.values():
        for other in sorted(ids)[1:]:
            link(
                sorted(ids)[0],
                other,
                "nominal_scene",
                "filename heuristic; not verified scene ground truth",
            )
    for _, block in rows.groupby("archivo_video", sort=True):
        entries = block.sort_values(["segundo", "frame_id"]).to_dict("records")
        for a, b in zip(entries, entries[1:], strict=False):
            if float(b["segundo"]) - float(a["segundo"]) <= window:
                link(
                    a["frame_id"],
                    b["frame_id"],
                    "temporal_window",
                    f"provider seconds; <= {window}; transitive closure",
                )
    for a, b, kind in visual_edges:
        if a not in graph.parent or b not in graph.parent:
            raise ValueError("Visual edge outside dataset")
        link(a, b, "visual_threshold", kind)
    groups = defaultdict(list)
    for value in sorted(graph.parent):
        groups[graph.find(value)].append(value)
    mapping = {
        v: digest("\n".join(ids).encode())[:24] for ids in groups.values() for v in ids
    }
    quarantine = set()
    for column in ("content_id", "source_frame_key"):
        for _, block in rows.groupby(column):
            if block.label_sha256.nunique() > 1:
                quarantine.update(mapping[v] for v in block.frame_id)
    return mapping, edges, quarantine


def visual_relations(rows, stores: dict, thresholds: dict):
    edges, bindings = [], {}
    if set(stores) != {"dinov2", "clip"} or set(thresholds) != set(stores):
        raise ValueError("Both DINOv2 and CLIP stores/thresholds required")
    representatives = (
        rows.sort_values("frame_id")
        .drop_duplicates("content_id")
        .set_index("content_id")
        .frame_id.to_dict()
    )
    for encoder, directory in stores.items():
        result = verify_features_against_manifest(directory, rows)
        if not result["reproducible_full_dataset_valid"]:
            raise ValueError("Feature store verification failed")
        meta = read_json(directory / "metadata.json")
        expected_model, expected_pooling = (
            ("facebook/dinov2-small", "cls_token")
            if encoder == "dinov2"
            else ("openai/clip-vit-base-patch32", "projected_pooler_output")
        )
        if (
            meta.get("model_id") != expected_model
            or meta.get("pooling_strategy") != expected_pooling
        ):
            raise ValueError("Unexpected mathematical encoder representation")
        index = pd.read_parquet(directory / "content_index.parquet")
        array = np.load(directory / "embeddings_l2.npy", allow_pickle=False)
        expected_dim = 384 if encoder == "dinov2" else 512
        if (
            array.shape != (len(index), expected_dim)
            or not np.isfinite(array).all()
            or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5)
            or set(index.content_id) != set(representatives)
            or not index.content_id.is_unique
        ):
            raise ValueError("Incomplete or invalid unique-content feature space")
        threshold = thresholds[encoder]
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Invalid cosine threshold")
        sim = (
            np.asarray(array, dtype=np.float64) @ np.asarray(array, dtype=np.float64).T
        )
        for i, j in zip(*np.where(np.triu(sim >= threshold, 1)), strict=True):
            edges.append(
                (
                    representatives[index.iloc[i].content_id],
                    representatives[index.iloc[j].content_id],
                    f"{encoder} cosine >= {threshold}",
                )
            )
        bindings[encoder] = {
            "feature_space_id": meta["feature_space_id"],
            "dataset_id": meta["dataset_id"],
            "files": {
                name: file_sha256(directory / name)
                for name in (
                    "metadata.json",
                    "embeddings_raw.npy",
                    "embeddings_l2.npy",
                    "content_index.parquet",
                    "record_index.parquet",
                )
            },
        }
    return edges, bindings


def verify_memberships(rows, edges):
    if (
        not rows.frame_id.is_unique
        or rows.new_split.isna().any()
        or not rows.new_split.isin(["train", "val", "test", "quarantine"]).all()
    ):
        raise ValueError("Invalid or incomplete assignments")
    for column in ("group_id", "content_id", "source_frame_key"):
        if rows.groupby(column).new_split.nunique().gt(1).any():
            raise ValueError(f"Cross-split identity: {column}")
    mapping = rows.set_index("frame_id").new_split.to_dict()
    for edge in edges:
        if mapping[edge["left"]] != mapping[edge["right"]]:
            raise ValueError("Cross-split registered dependency")
    return {
        "cross_split_exact": 0,
        "cross_split_registered_relations": 0,
        "independence_proven": False,
        "relations_checked": len(edges),
    }


def build(dataset: Path, config_path: Path, output: Path, stores: dict | None = None):
    rows = load_manifest(dataset)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config["protocol"] != "denoised_dependency_candidate_v1"
        or config["protected_test_anchor_mode"] != "none"
    ):
        raise ValueError(
            "Only a new candidate is supported; protected anchoring requires explicit verified correspondence"
        )
    edges, bindings = (
        visual_relations(rows, stores, config["cosine_thresholds"])
        if stores
        else ([], {})
    )
    mapping, relations, quarantine = dependency_graph(
        rows, config["temporal_window_seconds"], edges
    )
    rows["group_id"] = rows.frame_id.map(mapping)
    blockers = (
        []
        if stores
        else [
            "DINOv2 and CLIP stores for this denoised dataset are unavailable; visual constraints not evaluated"
        ]
    )
    report = {
        "state": "BLOCKED" if blockers else "CANDIDATE_NOT_FROZEN",
        "blockers": blockers,
        "dataset_id": dataset_id_from_manifest(rows),
        "groups": rows.group_id.nunique(),
        "quarantine_groups": sorted(quarantine),
        "relations_by_kind": pd.Series([r["kind"] for r in relations], dtype=str)
        .value_counts()
        .to_dict(),
        "configuration": config,
        "features": bindings,
        "c10_original": False,
        "protected_test_anchored": False,
        "limits": [
            "New transductive sensitivity policy, not C10 or a prospective independent test",
            "Nominal scenes/time constrain conservatively; provider provenance not independently verified",
            "No cross-edge leakage does not prove absolute independence",
            "No scientific approval/freeze created",
        ],
    }
    with transaction(output, dataset, config_path, *(stores or {}).values()) as stage:
        shutil.copy2(config_path, stage / "configuration.yaml")
        write_csv(
            stage / "relations.csv", relations, ["left", "right", "kind", "evidence"]
        )
        if not blockers:
            included = rows.loc[~rows.group_id.isin(quarantine)].copy()
            included["record_count"] = 1
            included["empty_annotation_count"] = (
                included[list(CLASS_COLUMNS)].sum(axis=1).eq(0).astype(int)
            )
            units = (
                included.groupby("group_id")[list(BALANCE_COLUMNS)].sum().reset_index()
            )
            units["content_count"] = (
                included.groupby("group_id")
                .content_id.nunique()
                .reindex(units.group_id)
                .to_numpy()
            )
            try:
                allocation, solver = milp_assignment(
                    units,
                    np.asarray(config["ratios"]),
                    config["seed"],
                    SplitConfig(node_limit=config["milp_node_limit"]),
                    require_class_coverage=True,
                )
            except ValueError as exc:
                write_json(
                    stage / "solver_failure.json",
                    {
                        "error": str(exc),
                        "evidence": getattr(
                            exc, "evidence", getattr(exc, "solver_result", None)
                        ),
                        "heuristic_substituted": False,
                    },
                )
                raise
            rows["new_split"] = rows.group_id.map(allocation).fillna("quarantine")
            report["invariants"] = verify_memberships(rows, relations)
            report["solver"] = solver
            report["counts"] = rows.new_split.value_counts().to_dict()
            report["class_counts"] = (
                rows.groupby("new_split")[list(CLASS_COLUMNS)].sum().to_dict("index")
            )
            for row in rows.to_dict("records"):
                for kind, column, expected in (
                    ("images", "image_path", row["content_id"]),
                    ("labels", "label_path", row["label_sha256"]),
                ):
                    target = stage / kind / row["new_split"] / Path(row[column]).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(dataset / row[column], target)
                    if file_sha256(target) != expected:
                        raise ValueError("Materialized copy differs")
            # No absolute path: Ultralytics resolves an absent path relative to YAML.
            (stage / "dataset.yaml").write_text(
                yaml.safe_dump(
                    {
                        "train": "images/train",
                        "val": "images/val",
                        "test": "images/test",
                        "nc": 5,
                        "names": dict(
                            enumerate(
                                ["vehicle", "building", "road", "river", "maquinaria"]
                            )
                        ),
                    }
                ),
                encoding="utf-8",
            )
            shutil.copytree(dataset / "metadata_original", stage / "metadata_original")
        else:
            rows["new_split"] = "UNASSIGNED"
        write_csv(stage / "memberships.csv", rows.to_dict("records"))
        write_json(stage / "audit.json", report)
        write_json(
            stage / "provenance.json",
            {
                "source_receipt_sha256": file_sha256(dataset / "receipt.json"),
                "config_sha256": file_sha256(config_path),
                "features": bindings,
            },
        )
        seal(stage, {"state": report["state"], "split_created": not bool(blockers)})
    return report
