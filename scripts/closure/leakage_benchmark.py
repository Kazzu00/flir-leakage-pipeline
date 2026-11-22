"""Controlled evaluation fixtures for the frozen general pipeline, not detector evidence.

Synthetic relationship truth is declared by construction before either policy
runs. No outputs are used to select seeds, ratios, relations or fixture families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance

from flir_pipeline.general.analysis import analyze
from flir_pipeline.general.contract import Evidence, Manifest, Relation, digest
from flir_pipeline.general.ingestion import ingest
from flir_pipeline.general.localization import localize
from flir_pipeline.general.publication import build, read_json, verify, write_json
from flir_pipeline.splitting.base import SplitConfig

PROTOCOL = {
    "schema": "controlled_leakage_benchmark_v1",
    "seeds": [0, 1, 2, 3, 4],
    "ratios": [0.6, 0.2, 0.2],
    "node_limit": 32,
    "mip_rel_gap": 0.001,
    "cases": ["visual_transforms", "temporal_repetitions", "candidates_only"],
    "family_count": 12,
    "members_per_family": 3,
    "duplicate_occurrences": 6,
    "generation_seed": 9182,
    "accepted_rule": "Two consecutive true links per family in the first two cases; none in candidates_only",
    "false_candidate_rule": "One link between first members of consecutive distinct families, circular",
    "missing_coverage_rule": "First occurrence in family zero reserved in all cases; common eligibility determined by frozen engine",
    "truth_scope": "Constructed relationships, not empirical scene semantics or calibrated visual similarity",
    "detector_training": False,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(root: Path, case: str):
    root.mkdir(parents=True)
    rng = np.random.default_rng(PROTOCOL["generation_seed"])
    inputs, families = [], []
    count = PROTOCOL["family_count"]
    for family in range(count):
        rgb = rng.integers(30, 180, size=(32, 40, 3), dtype=np.uint8)
        base = Image.fromarray(rgb)
        ImageDraw.Draw(base).rectangle((5, 6, 18, 17), fill=(230, family * 13, 50))
        variants = [
            base,
            ImageEnhance.Brightness(base).enhance(0.7),
            base.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ]
        if case == "temporal_repetitions":
            variants = [
                Image.fromarray(np.roll(np.array(base), step, axis=1))
                for step in range(3)
            ]
        members = []
        for j, image in enumerate(variants):
            frame_id = f"family-{family:02}-observation-{j}"
            filename = frame_id + ".png"
            image.save(root / filename)
            members.append(frame_id)
            inputs.append(
                {
                    "frame_id": frame_id,
                    "image": filename,
                    "original_split": ("train", "val", "test")[j],
                    "exclusion_reasons": ["synthetic_reserved_unknown_coverage"]
                    if family == j == 0
                    else [],
                }
            )
        if family < PROTOCOL["duplicate_occurrences"]:
            inputs.append(
                {
                    "frame_id": f"repeat-occurrence-{family}",
                    "image": members[0] + ".png",
                    "original_split": "test",
                }
            )
        families.append(members)
    videos = []
    if case == "temporal_repetitions":
        import av

        # Several independent constructed scenes share each video; video identity
        # alone must never determine grouping. Every scene's first frame recurs.
        for v in range(3):
            path = root / f"source-{v}.avi"
            with av.open(str(path), mode="w") as out:
                stream = out.add_stream("ffv1", rate=2)
                stream.width, stream.height, stream.pix_fmt = 40, 32, "bgr0"
                images = [
                    fid
                    for family in families[v * 4 : (v + 1) * 4]
                    for fid in [*family, family[0]]
                ]
                for i, fid in enumerate(images):
                    with Image.open(root / f"{fid}.png") as image:
                        frame = av.VideoFrame.from_ndarray(
                            np.array(image), format="rgb24"
                        )
                    frame.pts, frame.time_base = i, Fraction(1, 2)
                    for packet in stream.encode(frame):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            videos.append(
                {"video_id": f"source-{v}", "path": path.name, "sha256": sha(path)}
            )
    config_path = root / "input.yaml"
    config_path.write_text(
        yaml.safe_dump({"images": inputs, "classes": [], "videos": videos}),
        encoding="utf-8",
    )
    manifest = ingest(config_path, root)
    if videos:
        manifest = localize(manifest, root, root / "operations/localization.json")
    by_frame = {r.frame_id: r for r in manifest.records}
    truth = {
        "families": families,
        "true_edges": [],
        "false_candidates": [],
        "protocol_sha256": digest(PROTOCOL),
        "synthetic_only": True,
    }
    for i, members in enumerate(families):
        truth["true_edges"].extend([[members[0], members[1]], [members[1], members[2]]])
        truth["false_candidates"].append([members[0], families[(i + 1) % count][0]])
    write_json(root / "truth.json", truth)
    ev = Evidence(
        path="truth.json",
        sha256=sha(root / "truth.json"),
        method="explicit_synthetic_generator",
        version="1",
        provenance="Controlled fixture, not human FAC review",
        reviewer="synthetic_protocol" if case != "candidates_only" else None,
        reviewed_at="2026-11-06T00:00:00Z" if case != "candidates_only" else None,
    )
    edges = []
    for i, (a, b) in enumerate(truth["true_edges"]):
        accepted = case != "candidates_only"
        edges.append(
            Relation(
                relation_id=f"true-{i:02}",
                left=by_frame[a].content_id,
                right=by_frame[b].content_id,
                kind="temporal" if case == "temporal_repetitions" else "visual",
                status="confirmed" if accepted else "candidate",
                accepted=accepted,
                evidence=ev,
            )
        )
    for i, (a, b) in enumerate(truth["false_candidates"]):
        edges.append(
            Relation(
                relation_id=f"false-{i:02}",
                left=by_frame[a].content_id,
                right=by_frame[b].content_id,
                kind="visual",
                status="candidate",
                accepted=False,
                score=0.99,
                evidence=ev,
            )
        )
    # A separate unresolved judgment stays ambiguous despite a high nominal score.
    edges.append(
        Relation(
            relation_id="ambiguous-control",
            left=by_frame[families[2][0]].content_id,
            right=by_frame[families[7][0]].content_id,
            kind="external",
            status="ambiguous",
            accepted=False,
            evidence=ev,
        )
    )
    manifest = Manifest.model_validate({**manifest.model_dump(), "relations": edges})
    write_json(root / "manifest.json", manifest.canonical())
    return manifest, truth


def independent_edge_counts(edges, assignments):
    """Evaluate generator occurrence pairs directly, without reading graph closure."""
    covered = [(a, b) for a, b in edges if a in assignments and b in assignments]
    return {
        "crossing": sum(assignments[a] != assignments[b] for a, b in covered),
        "evaluable": len(covered),
        "unknown": len(edges) - len(covered),
        "total": len(edges),
    }


def benchmark(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", PROTOCOL)
    config = SplitConfig(
        target_ratios=tuple(PROTOCOL["ratios"]),
        seeds=tuple(PROTOCOL["seeds"]),
        node_limit=PROTOCOL["node_limit"],
        mip_rel_gap=PROTOCOL["mip_rel_gap"],
    )
    write_json(output / "effective_config.json", asdict(config))
    summary = []
    for case in PROTOCOL["cases"]:
        root = output / case
        manifest, truth = generate(root / "inputs", case)
        build(manifest, config, root / "inputs", root / "publication")
        replay = verify(root / "publication", root / "inputs")
        write_json(root / "replay.json", replay)
        result = read_json(root / "publication/analysis.json")
        historical = independent_edge_counts(
            truth["true_edges"], result["historical_assignments"]
        )
        assert historical["crossing"] == historical["total"]
        assert (
            result["historical_diagnostics"]["exact_duplicates"][
                "crossing_content_groups"
            ]
            == 6
        )
        no_candidates = Manifest.model_validate(
            {
                **manifest.model_dump(),
                "relations": [r.model_dump() for r in manifest.relations if r.accepted],
            }
        )
        control = analyze(no_candidates, config)
        reordered = Manifest.model_validate(
            {
                **manifest.model_dump(),
                "records": list(reversed(manifest.records)),
                "relations": list(reversed(manifest.relations)),
            }
        )
        permuted = analyze(reordered, config)
        assert digest(permuted) == digest(result)
        rows = []
        for index, run in enumerate(result["runs"]):
            assert run["state"] == "assigned"
            assert run["assignments"] == control["runs"][index]["assignments"]
            observed = independent_edge_counts(truth["true_edges"], run["assignments"])
            false = independent_edge_counts(
                truth["false_candidates"], run["assignments"]
            )
            assert (
                run["diagnostics"]["exact_duplicates"]["crossing_content_groups"] == 0
            )
            if case != "candidates_only":
                assert observed == {
                    k: run["diagnostics"]["relations"]["accepted"][k] for k in observed
                }
                assert observed["unknown"] > 0
                if run["policy"] == "dependency_group_balanced":
                    assert observed["crossing"] == 0
            rows.append(
                {
                    "policy": run["policy"],
                    "seed": run["seed"],
                    "constructed_true_relations": observed,
                    "false_candidates": false,
                    "accepted": run["diagnostics"]["relations"]["accepted"],
                    "ambiguous": run["diagnostics"]["relations"]["ambiguous"],
                    "membership_sha256": run["membership_sha256"],
                    "solver": run["solver"],
                }
            )
        if case == "candidates_only":
            assert all(r["memberships_equal"] for r in result["equivalence"])
        else:
            # This asserts detection sensitivity in the preregistered fixture;
            # failure is reported, never repaired by searching seeds/relations.
            assert any(
                r["constructed_true_relations"]["crossing"] > 0 for r in rows[:5]
            )
        report = {
            "case": case,
            "historical_constructed_true_relations": historical,
            "historical_exact_crossings": 6,
            "rows": rows,
            "equivalence": result["equivalence"],
            "candidate_removal_preserves_memberships": True,
            "shuffled_input_full_replay_equal": True,
            "replay_verified": replay["verified"],
            "temporal_coverage": result["temporal_coverage"],
            "experimental_scope": "Constructed leakage relations; no detector accuracy inference",
        }
        write_json(root / "benchmark.json", report)
        summary.append(report)
    write_json(output / "benchmark_summary.json", summary)
    print(
        json.dumps(
            [
                {
                    "case": r["case"],
                    "exact_policy_true_crossings": [
                        v["constructed_true_relations"]["crossing"]
                        for v in r["rows"][:5]
                    ],
                    "grouped_true_crossings": [
                        v["constructed_true_relations"]["crossing"]
                        for v in r["rows"][5:]
                    ],
                }
                for r in summary
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    benchmark(parser.parse_args().output)
