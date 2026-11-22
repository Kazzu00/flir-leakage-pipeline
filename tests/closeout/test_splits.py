import numpy as np
import pandas as pd
import pytest

from flir_pipeline.closeout.splits import dependency_graph, verify_memberships
from flir_pipeline.splitting.base import SplitConfig
from flir_pipeline.splitting.construction import AssignmentFailure, milp_assignment


def rows():
    return pd.DataFrame(
        [
            {
                "frame_id": str(i),
                "content_id": f"content-{i}",
                "source_frame_key": f"source:{i}",
                "label_sha256": "label",
                "nombre_original": f"frame{i}.jpg",
                "archivo_video": "one_video",
                "dataset_origen": "source",
                "segundo": float(i * 10),
            }
            for i in range(18)
        ]
    )


def test_one_video_is_not_an_atomic_unit():
    mapping, edges, _ = dependency_graph(rows(), 2)
    assert len(set(mapping.values())) == 18
    assert not edges


def test_exact_source_scene_time_visual_closure_and_quarantine():
    frame = rows()
    frame.loc[1, "content_id"] = frame.loc[0, "content_id"]
    frame.loc[2, "source_frame_key"] = frame.loc[1, "source_frame_key"]
    frame.loc[3:4, "nombre_original"] = "frame_scene123.jpg"
    frame.loc[5, "segundo"] = 41
    frame.loc[0, "label_sha256"] = "conflicting"
    mapping, edges, quarantine = dependency_graph(frame, 2, [("2", "3", "dinov2")])
    assert len({mapping[str(i)] for i in range(6)}) == 1
    assert quarantine == {mapping["0"]}
    frame["group_id"] = frame.frame_id.map(mapping)
    frame["new_split"] = frame.group_id.map(
        lambda v: "quarantine" if v in quarantine else "train"
    )
    assert verify_memberships(frame, edges)["cross_split_registered_relations"] == 0
    frame.loc[1, "new_split"] = "test"
    with pytest.raises(ValueError, match="Cross-split"):
        verify_memberships(frame, edges)


def test_order_invariant_graph():
    frame = rows()
    a = dependency_graph(frame, 11)[0]
    b = dependency_graph(frame.sample(frac=1, random_state=42), 11)[0]
    assert a == b


def test_milp_balance_and_indivisibility():
    frame = rows()
    mapping, edges, _ = dependency_graph(frame, 2, [("0", "1", "clip")])
    frame["group_id"] = frame.frame_id.map(mapping)
    frame["record_count"] = 1
    frame["empty_annotation_count"] = 0
    for i in range(5):
        frame[f"class_{i}"] = 1
    cols = ["record_count", *[f"class_{i}" for i in range(5)], "empty_annotation_count"]
    units = frame.groupby("group_id")[cols].sum().reset_index()
    allocation, result = milp_assignment(
        units,
        np.array([0.7, 0.15, 0.15]),
        42,
        SplitConfig(),
        require_class_coverage=True,
    )
    frame["new_split"] = frame.group_id.map(allocation)
    assert set(frame.new_split) == {"train", "val", "test"}
    assert result["primal_feasibility_verified"]
    verify_memberships(frame, edges)
    with pytest.raises(AssignmentFailure):
        milp_assignment(
            units.head(2),
            np.array([0.7, 0.15, 0.15]),
            42,
            SplitConfig(),
            require_class_coverage=True,
        )


@pytest.mark.parametrize("solver_fails", [False, True])
def test_end_to_end_candidate_materialization(tmp_path, monkeypatch, solver_fails):
    import csv
    import io

    import yaml
    from test_import_denoised import fixture_members, make_tar

    from flir_pipeline.closeout import splits
    from flir_pipeline.closeout.denoised import prepare
    from flir_pipeline.closeout.io import read_json, verify

    members = []
    for name, raw in fixture_members(18):
        if "/labels/" in name:
            raw = "".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in range(5)).encode()
        if name.endswith("correspondencia.csv"):
            reader = csv.DictReader(io.StringIO(raw.decode()))
            records = list(reader)
            for row in records:
                for key in (
                    "vehicle",
                    "building",
                    "road",
                    "river",
                    "SDZI",
                    "maquinaria",
                ):
                    if key in row:
                        row[key] = "1"
                row["n_cajas"] = "5"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(records)
            raw = output.getvalue().encode()
        members.append((name, raw))
    dataset = tmp_path / "dataset"
    prepare(make_tar(tmp_path / "source.tar", members), dataset, 18, True)
    config = {
        "protocol": "denoised_dependency_candidate_v1",
        "protected_test_anchor_mode": "none",
        "temporal_window_seconds": 0,
        "cosine_thresholds": {"dinov2": 0.98, "clip": 0.99},
        "ratios": [0.7, 0.15, 0.15],
        "seed": 42,
        "milp_node_limit": 32,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    # Numerical model extraction is covered by native feature tests; fixture edges
    # exercise propagation, solver, materialization and receipt verification here.
    monkeypatch.setattr(
        splits,
        "visual_relations",
        lambda frame, *args: (
            [(frame.iloc[0].frame_id, frame.iloc[1].frame_id, "synthetic cosine edge")],
            {"synthetic": True},
        ),
    )
    target = tmp_path / "split"
    if solver_fails:

        def fail(*args, **kwargs):
            raise AssignmentFailure("Synthetic infeasible graph", {"status": 2})

        monkeypatch.setattr(splits, "milp_assignment", fail)
        with pytest.raises(AssignmentFailure, match="infeasible"):
            splits.build(
                dataset,
                config_path,
                target,
                {"dinov2": tmp_path / "dino", "clip": tmp_path / "clip"},
            )
        assert not target.exists()
        (stage,) = tmp_path.glob(".split.partial-*")
        assert len(pd.read_csv(stage / "preassignment_groups.csv")) == 18
        assert read_json(stage / "preassignment.json")["groups"] == 17
        assert read_json(stage / "solver_failure.json")["evidence"] == {"status": 2}
        assert not (stage / "dataset.yaml").exists()
        return
    report = splits.build(
        dataset,
        config_path,
        target,
        {"dinov2": tmp_path / "dino", "clip": tmp_path / "clip"},
    )
    assert report["state"] == "CANDIDATE_NOT_FROZEN"
    assert report["invariants"]["cross_split_registered_relations"] == 0
    assert sum(report["counts"].values()) == 18
    assert report["c10_original"] is False
    assert read_json(target / "receipt.json")["split_created"] is True
    verify(target)
    assert len(list((target / "images").rglob("*.png"))) == 18
    assert len(list((target / "labels").rglob("*.txt"))) == 18
