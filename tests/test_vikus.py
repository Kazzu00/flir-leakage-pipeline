"""Offline synthetic collection contracts; no upstream download or model fitting."""

import json
from dataclasses import replace
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_explorer import experiment as experiment
from test_explorer import publish

from flir_pipeline.explorer.data import load_cluster, load_split
from flir_pipeline.explorer.discovery import read_json, sha256
from flir_pipeline.explorer.vikus import (
    annotation_metadata,
    build_bundle,
    create_images,
    find_source_reduction,
    layout_table,
    metadata_table,
    safe_bundle_path,
    verify_bundle,
    viewer_config,
)
from flir_pipeline.explorer.vikus_server import local_server
from flir_pipeline.explorer.vikus_upstream import install_runtime, replace_once


def loaded(experiment, strategy="cluster_aware"):
    _, _, manifest, run, splits = experiment
    return load_cluster(run, manifest), load_split(next(s for s in splits if s.strategy == strategy), manifest)


def reduction(experiment, *, method="pacmap"):
    root, _, _, _, _ = experiment
    cluster, _ = loaded(experiment)
    # Deliberately reverse content order: joining by row position is incorrect.
    index = cluster.contents[["content_id", "embedding_row"]].iloc[::-1].reset_index(drop=True)
    index.embedding_row = np.arange(len(index))
    coordinates = np.array([[i / 7, (5 - i) * 3.7] for i in range(len(index))], dtype=np.float64)
    directory = root / "reductions" / method
    publish(directory, {"artifact_kind": "reduction_run", "method": method, "N": len(index),
                        "dataset_id": cluster.run.dataset_id, "feature_space_id": "features",
                        "extractor": cluster.run.encoder, "output_dimension": 2, "reduction_space_id": "reduced"},
            {"content_index.parquet": index}, {"coordinates.npy": coordinates})
    signatures = {n: sha256(directory / n) for n in ("coordinates.npy", "content_index.parquet", "metadata.json")}
    meta = {**cluster.run.metadata, "feature_space_id": "features", "reduction_space_id": "reduced",
            "input_signatures": {"reduction": signatures}}
    return replace(cluster, run=replace(cluster.run, metadata=meta)), directory, index, coordinates


def test_metadata_preserves_occurrences_noise_groups_and_year(experiment):
    _, _, manifest, _, _ = experiment
    cluster, split = loaded(experiment)
    table = metadata_table(cluster, split, manifest)
    assert len(table) == table.id.nunique() == 6
    assert table._occurrence_count.sum() == 8
    occurrences = [r for value in table._frame_occurrences for r in json.loads(value)]
    assert {r["frame_id"] for r in occurrences} == set(manifest.frame_id)
    assert table.year.eq(0).all()
    noise = table.loc[table._noise.eq("true")].iloc[0]
    assert noise._cluster_id == -1 and noise._cluster_group == "Noise (-1)"
    assert noise._group_id == "noise-4" and noise._new_split == "test"
    assert set(table._split_group) == {"train", "validation", "test"}
    assert table.loc[table._sequence_id.eq("A"), "_frame_index"].tolist() == [1, 3, 40]
    assert table.loc[table._sequence_id.eq("C"), "_frame_index"].item() == ""
    assert table._sequence_group.str.startswith("synthetic.zip / ").all()
    assert table._empty_annotation.eq("unknown").all()
    config = viewer_config(table, cluster, split, [{"method": "pacmap", "url": "data/layouts/pacmap.csv"}])
    assert [v["title"] for v in config["loader"]["layouts"]] == ["Clusters", "Sequences", "Cluster-aware split", "PaCMAP visual similarity"]
    assert config["sortArrays"]["_cluster_group"] == ["Cluster 0", "Cluster 1", "Noise (-1)"]
    assert len(config["filter"]["dimensions"]) == 5
    assert all(v.get("groupKey") != "year" for v in config["loader"]["layouts"])
    assert all(v["source"] != "year" for v in config["detail"]["structure"])


def test_historical_membership_is_not_collapsed(experiment):
    cluster, split = loaded(experiment, "historical")
    table = metadata_table(cluster, split, experiment[2])
    row = table.set_index("id").loc[experiment[2].iloc[0].content_id]
    assert json.loads(row._split_memberships) == ["train", "test"]
    assert row._split_group == "train + test"
    assert json.loads(row._original_split_membership) == ["test", "train"]
    assert metadata_table(cluster, None, experiment[2])._new_split.eq("Unassigned").all()


def test_posthoc_annotations_preserve_conflict_and_unknown():
    annotations = pd.DataFrame({"classes_present": ["0|4", ""], "label_sha256": ["a", "b"]})
    value = annotation_metadata(annotations)
    assert value["_contains_vehicles"] == value["_contains_heavy_machinery"] == "true"
    assert value["_contains_rivers"] == "false"
    assert value["_empty_annotation"] == "mixed" and value["_annotation_conflict"] == "true"
    assert set(json.loads(value["_class_presence"])) == {"Vehicles", "Heavy Machinery", "Empty annotation"}
    annotations.loc[1, "classes_present"] = None
    value = annotation_metadata(annotations)
    assert value["_empty_annotation"] == value["_contains_rivers"] == "unknown"
    assert value["_contains_vehicles"] == "true"


def test_manifest_provenance_cannot_change_after_loading(experiment):
    cluster, split = loaded(experiment)
    changed = experiment[2].copy()
    changed.loc[0, "possible_sequence"] = "Different"
    with pytest.raises(ValueError, match="provenance mapping"):
        metadata_table(cluster, split, changed)
    changed.loc[0, "label_sha256"] = "Different"
    with pytest.raises(ValueError, match="dataset identities"):
        metadata_table(cluster, split, changed)


def test_layout_alignment_and_exact_csv_roundtrip(experiment, tmp_path):
    cluster, directory, index, coordinates = reduction(experiment)
    rows, info = layout_table(cluster, directory)
    expected = pd.DataFrame(coordinates, index=index.content_id, columns=["x", "y"]).loc[rows.id]
    np.testing.assert_array_equal(rows[["x", "y"]], expected)
    path = tmp_path / "display.csv"
    rows.to_csv(path, index=False, float_format="%.17g")
    np.testing.assert_array_equal(pd.read_csv(path, float_precision="round_trip")[["x", "y"]], expected)
    assert info["source_of_clustering"] and info["csv_transform"].startswith("none")
    assert find_source_reduction(cluster, directory.parent) == directory


@pytest.mark.parametrize("corruption", ["feature", "dataset", "source", "checksum", "duplicate", "nonfinite"])
def test_reduction_rejects_misalignment_and_corruption(experiment, corruption):
    cluster, directory, index, coordinates = reduction(experiment)
    meta = read_json(directory / "metadata.json")
    if corruption in {"feature", "dataset", "source"}:
        field = {"feature": "feature_space_id", "dataset": "dataset_id", "source": "reduction_space_id"}[corruption]
        meta[field] = "different"
    elif corruption == "duplicate":
        index.loc[0, "content_id"] = index.loc[1, "content_id"]
        index.to_parquet(directory / "content_index.parquet", index=False)
        meta["output_sha256"]["content_index.parquet"] = sha256(directory / "content_index.parquet")
    else:
        coordinates[0, 0] = np.nan if corruption == "nonfinite" else 9
        np.save(directory / "coordinates.npy", coordinates)
        if corruption == "nonfinite":
            meta["output_sha256"]["coordinates.npy"] = sha256(directory / "coordinates.npy")
    (directory / "metadata.json").write_text(json.dumps(meta))
    if corruption in {"duplicate", "nonfinite"}:
        # These invalid arrays are rejected even when presented as intact extras.
        with pytest.raises(ValueError, match="coverage"):
            layout_table(cluster, directory, source=False)
    else:
        with pytest.raises(ValueError):
            layout_table(cluster, directory)


def test_extra_tsne_requires_same_feature_space(experiment):
    cluster, directory, _, _ = reduction(experiment, method="tsne")
    _, info = layout_table(cluster, directory, source=False)
    assert info["method"] == "tsne" and not info["source_of_clustering"]


def test_images_sprites_mapping_and_source_readonly(experiment):
    root, source, manifest, _, _ = experiment
    cluster, split = loaded(experiment)
    table = metadata_table(cluster, split, manifest)
    before = sha256(source / "synthetic.zip")
    output = safe_bundle_path(root, "synthetic", source)
    assets = create_images(output, table, manifest, source)
    assert assets["images"] == 6 and assets["sprite_sheets"] == 1
    sprites = read_json(output / "data/sprites/manifest.json")["spritesheets"][0]["sprites"]
    assert {s["name"] for s in sprites} == set(table.id)
    for sprite in sprites:
        assert sprite["dimension"] == {"w": 24, "h": 16}
        assert sprite["position"]["x"] + 24 <= 2048
        with Image.open(output / "data/images" / f"{sprite['name']}.jpg") as image:
            assert image.size == (1024, 683)
    assert {p.stem for p in (output / "data/images").iterdir()} == set(table.id)
    assert sha256(source / "synthetic.zip") == before


@pytest.mark.parametrize("name", ["../escape", "a/b", "C:\\outside", ".", "", "x:y"])
def test_safe_output_rejects_nonlocal_names(experiment, name):
    root, source, *_ = experiment
    with pytest.raises(ValueError):
        safe_bundle_path(root, name, source)


def test_safe_output_rejects_source_tree_overlap(experiment):
    root, *_ = experiment
    with pytest.raises(ValueError):
        safe_bundle_path(root, "synthetic", root)


def test_upstream_checksums_and_patch_context_are_mandatory(tmp_path):
    archive = tmp_path / "not-upstream.zip"
    archive.write_bytes(b"not the pinned upstream")
    with pytest.raises(ValueError, match="pinned upstream"):
        install_runtime(archive, tmp_path / "output")
    with pytest.raises(ValueError, match="patch context"):
        replace_once("twice twice", "twice", "once")


def test_build_is_offline_idempotent_readonly_and_server_is_scoped(experiment, monkeypatch):
    root, source, manifest, _, _ = experiment
    cluster, split = loaded(experiment)
    manifest_path = root / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    original = {p: sha256(p) for p in root.rglob("*") if p.is_file()}

    def minimal_runtime(_, destination):
        (destination / "index.html").write_text("<!doctype html><title>Synthetic</title>")
        return {"synthetic": True}

    monkeypatch.setattr("flir_pipeline.explorer.vikus.install_runtime", minimal_runtime)
    output = build_bundle(root, manifest_path, source, cluster, split, [], root / "unused.zip")
    receipt = verify_bundle(output)
    assert receipt["sources_unchanged"] and not receipt["scientific_experiments_executed"]
    assert receipt["content_count"] == 6 and receipt["occurrence_count"] == 8
    assert all(sha256(p) == checksum for p, checksum in original.items())
    assert build_bundle(root, manifest_path, source, cluster, split, [], root / "unused.zip") == output
    table = pd.read_csv(output / "data/data.csv", keep_default_na=False)
    assert len(table) == 6 and table.id.is_unique
    assert {"id", "keywords", "_cluster_id", "_frame_index", "_noise", "_contains_roads"} <= set(table)
    with local_server(output, root, 0) as server:
        assert server.server_address[0] == "127.0.0.1"

        class InMemoryRequest:
            """Exercise real HTTP parsing/dispatch without a network connection."""

            def __init__(self, path, method="GET", host="127.0.0.1"):
                self.input = BytesIO(f"{method} {path} HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
                self.output = bytearray()

            def makefile(self, *_):
                return self.input

            def sendall(self, value):
                self.output.extend(value)

        def response(path, **kwargs):
            request = InMemoryRequest(path, **kwargs)
            server.RequestHandlerClass(request, ("127.0.0.1", 1), server)
            return bytes(request.output)

        assert b"200 OK" in response("/")
        assert b"Content-Security-Policy: default-src 'self'" in response("/")
        assert b"Content-type: image/jpeg" in response("/" + table._image_url[0])
        assert b"404" in response("/../manifest.parquet")
        assert b"404" in response("/data/")
        assert b"403" in response("/", host="untrusted.example")
        assert b"501" in response("/", method="POST")
    (output / "data/data.csv").write_text("tampered")
    with pytest.raises(ValueError, match="receipt"):
        verify_bundle(output)
