"""Immutable local reports, source checks, and read-only links to Human Audit v1.1."""

import csv
import html
import io
import json
import platform
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inventory import encoded, file_sha, inventory, read_image, safe_path, select, sha
from .matching import BudgetExceeded, Decoder, search
from .models import Case, Config, Evidence, HumanVerification


def audit_links(path, rows):
    """Index joins are navigation suggestions, never evidence of visual identity."""
    if path is None:
        return {}, None
    path = Path(path).resolve()
    before = file_sha(path)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        events = [
            json.loads(payload)
            for (payload,) in db.execute("SELECT payload FROM events")
        ]
    links = {}
    for row in rows:
        links[row["image_id"]] = [
            {
                "event_id": e["event_id"],
                "basis": "index_reference_candidate_only_not_visual_identity",
            }
            for e in events
            if e.get("start_index") is not None
            and e.get("end_index") is not None
            and e["start_index"] <= row["dataset_index"] <= e["end_index"]
        ]
    if file_sha(path) != before:
        raise ValueError("Audit ledger changed during read")
    return links, before


def sheet(path, image, top, frames):
    panels = [("Original image (split hidden)", np.asarray(image))]
    for match in top:
        panels.append(
            (
                f"{'Control' if match['search_stage'] == 'outside_window_control' else 'Candidate'} PTS={match['pts']} t={match['second']:.6f}",
                frames[match["pts"]]["rgb"],
            )
        )
    if top:
        best = top[0]["second"]
        for side in (-1, 1):
            options = [
                f for f in frames.values() if 0 < side * (f["second"] - best) <= 1
            ]
            if options:
                frame = min(options, key=lambda f: abs(abs(f["second"] - best) - 0.3))
                panels.append(
                    (
                        f"Neighbor PTS={frame['pts']} t={frame['second']:.6f}",
                        frame["rgb"],
                    )
                )
    canvas = Image.new("RGB", (1152, ((len(panels) + 2) // 3) * 250), "white")
    draw = ImageDraw.Draw(canvas)
    metadata = []
    for i, (label, rgb) in enumerate(panels):
        x, y = (i % 3) * 384, (i // 3) * 250
        canvas.paste(Image.fromarray(rgb).resize((384, 216)), (x, y + 24))
        draw.text((x + 4, y + 4), label, fill="black")
        metadata.append(label)
    canvas.save(path)
    return metadata


def csv_bytes(cases):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=["alignment_id", "image_id", "status", "case_json"]
    )
    writer.writeheader()
    for case in cases:
        writer.writerow(
            {
                **{k: case[k] for k in ("alignment_id", "image_id", "status")},
                "case_json": json.dumps(case, sort_keys=True),
            }
        )
    return stream.getvalue().encode("utf-8")


def make_report(output, evidence, manifest, selection):
    rows = {r["image_id"]: r for r in manifest["images"]}
    by_split = {}
    for split in ("train", "val", "test"):
        cases = [
            c
            for c in evidence["cases"]
            if rows[c["image_id"]]["original_split"] == split
        ]
        by_split[split] = {
            "available_original_images": sum(
                r["original_split"] == split
                and r["source_archive_or_path"] == selection["selection_archive"]
                and r["read_status"] == "readable"
                for r in rows.values()
            ),
            "sample": len(cases),
            **{
                s: sum(c["status"] == s for c in cases)
                for s in ("candidate", "ambiguous", "not_found")
            },
            "evidence_sheets": sum(c["evidence_locator"] is not None for c in cases),
            "errors": sum(bool(c["errors"]) for c in cases),
        }
    summary = {
        "by_split": by_split,
        "human_verified": 0,
        "manifest_occurrences": len(rows),
        "selected": len(selection["selected"]),
        "missing_diagnostic_indices": selection["missing_diagnostic_indices"],
        "budget_exhausted": any(
            "budget" in e.lower() for c in evidence["cases"] for e in c["errors"]
        ),
        "diagnostics": [
            {
                "index": rows[c["image_id"]]["dataset_index"],
                "split": rows[c["image_id"]]["original_split"],
                "status": c["status"],
                "alignment_id": c["alignment_id"],
                "best_pts_second": c["top_k"][0]["second"] if c["top_k"] else None,
                "offset_from_nominal_seconds": c["top_k"][0]["second"]
                - (
                    evidence["config"]["offset_hypothesis"]
                    + rows[c["image_id"]]["dataset_index"]
                )
                if c["top_k"]
                else None,
            }
            for c in evidence["cases"]
            if rows[c["image_id"]]["dataset_index"]
            in (199, 200, 349, 350, 399, 400, 499, 500, 501)
        ],
    }
    (output / "summary.json").write_bytes(encoded(summary))
    cards = []
    for number, case in enumerate(evidence["cases"], 1):
        image = (
            f'<img loading="lazy" width="1152" src="{case["evidence_locator"]}" alt="Original, candidates and temporal neighbors">'
            if case["evidence_locator"]
            else "No visual evidence available"
        )
        row = rows[case["image_id"]]
        if case.get("control_evidence_locator"):
            image += f'<details><summary>Outside-window controls (not truth labels)</summary><img loading="lazy" src="{case["control_evidence_locator"]}" alt="Documented outside-window controls"></details>'
        cards.append(
            f"<section><h2>Case {number}: {case['status']}</h2><p>{html.escape(case['reason'])}</p>{image}<details><summary>Provenance (open after first visual assessment)</summary><pre>{html.escape(json.dumps(row, indent=2))}</pre></details><details><summary>Scores, alternatives and controls (uncalibrated)</summary><pre>{html.escape(json.dumps(case, indent=2))}</pre></details></section>"
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>FLIR Phase 2A evidence</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}section{border-top:2px solid #777;padding:1em}</style><h1>Visual provenance candidates; human review pending</h1><p>No labels, no automatic verification, no independent sequences. Scores are retrieval diagnostics, not confidence. Outside-window controls are not ground-truth negatives of region identity. Split and index are hidden until provenance is opened. Review continuity and region dependence separately using the pilot protocol.</p>'
        + "".join(cards)
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return summary


def run(root, video, output, config=None, sample_only=False, audit_db=None):
    import av
    import cv2

    implementation = {
        p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    config = config or Config()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside original data root")
    video_path = safe_path(root, video)
    # Source stat receipts supplement hashes; no writes to source directories.
    source_stats = {
        p.relative_to(root).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in root.rglob("*")
        if p.is_file()
    }
    manifest = inventory(root, video)
    selection = select(manifest, config)
    file_sha_cached = file_sha(video_path)
    runtime = {
        "python": platform.python_version(),
        "av": av.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    identity = sha(
        encoded(
            [
                config.model_dump(),
                sha(encoded(manifest)),
                sha(encoded(selection)),
                file_sha_cached,
                implementation,
                runtime,
                file_sha(audit_db) if audit_db else None,
            ]
        )
    )
    if output.exists():
        if (output / "evidence.json").exists():
            existing = verify(output)
            if existing["run_id"] == identity:
                return json.loads((output / "summary.json").read_text(encoding="utf-8"))
        raise ValueError(
            "Output already exists; use a new run directory (no silent overwrite)"
        )
    output.mkdir(parents=True)
    (output / "manifest.json").write_bytes(encoded(manifest))
    (output / "selection.json").write_bytes(encoded(selection))
    (output / "config.json").write_bytes(encoded(config.model_dump()))
    if sample_only:
        return {
            "sample_only": True,
            "selected": len(selection["selected"]),
            "manifest_occurrences": len(manifest["images"]),
            "missing": selection["missing_diagnostic_indices"],
        }
    selected_ids = {r["image_id"] for r in selection["selected"]}
    rows = sorted(
        [r for r in manifest["images"] if r["image_id"] in selected_ids],
        key=lambda r: r["image_id"],
    )
    links, ledger_sha = audit_links(audit_db, rows)
    cases, exhausted = [], False
    decoder = Decoder(video_path, config.max_decoded_frames, config.max_seconds)
    (output / "sheets").mkdir()
    started = time.monotonic()
    try:
        for number, row in enumerate(rows, 1):
            aid = sha(encoded([identity, row["image_id"]]))
            case = dict(
                alignment_id=aid,
                image_id=row["image_id"],
                status="not_found",
                reason="No evidence processed",
                top_k=[],
                controls=[],
                searched_intervals=[],
                candidate_interval=None,
                evidence_locator=None,
                evidence_sha256=None,
                audit_event_links=links.get(row["image_id"], []),
                errors=[],
            )
            try:
                if exhausted:
                    raise BudgetExceeded(
                        "Run budget already exhausted; sample retained as unprocessed"
                    )
                _, image = read_image(root, row)
                result, frames = search(image, row["dataset_index"], decoder, config)
                case.update(result)
                if case["top_k"]:
                    locator = f"sheets/{aid}.jpg"
                    panels = sheet(output / locator, image, case["top_k"], frames)
                    (output / f"sheets/{aid}.json").write_bytes(
                        encoded(
                            {
                                "alignment_id": aid,
                                "image_id": row["image_id"],
                                "panels": panels,
                                "source_video_sha256": file_sha_cached,
                                "extraction": decoder.metadata,
                            }
                        )
                    )
                    case.update(
                        evidence_locator=locator,
                        evidence_sha256=file_sha(output / locator),
                    )
                    if case["controls"]:
                        control_locator = f"sheets/{aid}-controls.jpg"
                        sheet(output / control_locator, image, case["controls"], frames)
                        case.update(
                            control_evidence_locator=control_locator,
                            control_evidence_sha256=file_sha(output / control_locator),
                        )
            except (OSError, ValueError, RuntimeError) as exc:
                exhausted = exhausted or isinstance(exc, BudgetExceeded)
                case.update(
                    status="not_found",
                    reason="Processing incomplete; not evidence of absence",
                    errors=[str(exc)],
                )
            cases.append(Case(**case).model_dump())
            print(
                json.dumps(
                    {
                        "progress": f"{number}/{len(rows)}",
                        "status": case["status"],
                        "decoded_frames": decoder.decoded,
                    }
                ),
                flush=True,
            )
    finally:
        decoder.close()
    for relative, stat in source_stats.items():
        path = root / relative
        if [path.stat().st_size, path.stat().st_mtime_ns] != stat:
            raise ValueError("Source changed during alignment; unpublished run")
    if audit_db is not None and file_sha(audit_db) != ledger_sha:
        raise ValueError("Original audit ledger changed")
    if implementation != {
        p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }:
        raise ValueError("Implementation changed during run; evidence not published")
    evidence = Evidence(
        **{
            "schema": "alignment-evidence-v1",
            "run_id": identity,
            "config": config,
            "manifest_sha256": file_sha(output / "manifest.json"),
            "selection_sha256": file_sha(output / "selection.json"),
            "source_video": video,
            "source_video_sha256": file_sha(video_path),
            "video_metadata": decoder.metadata,
            "execution": {
                "implementation_sha256": implementation,
                "runtime": runtime,
                "python": platform.python_version(),
                "av": av.__version__,
                "opencv": cv2.__version__,
                "git_commit": commit,
                "decoded_frames": decoder.decoded,
                "sampled_frames": decoder.sampled,
                "elapsed_seconds": time.monotonic() - started,
                "audit_ledger_sha256_read_only": ledger_sha,
                "sources_stat_unchanged": True,
                "score_policy": "lexicographic support/inliers/ratio/correlation; not calibrated confidence",
                "device": "cpu",
            },
            "cases": cases,
        }
    ).model_dump(mode="json", by_alias=True)
    (output / "evidence.json").write_bytes(encoded(evidence))
    (output / "decode-log.json").write_bytes(encoded(decoder.seek_log))
    (output / "evidence.csv").write_bytes(csv_bytes(cases))
    summary = make_report(output, evidence, manifest, selection)
    receipts = {
        p.relative_to(output).as_posix(): file_sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "checksums.json").write_bytes(encoded(receipts))
    verify(output)
    return summary


def verify(output):
    output = Path(output)
    receipts = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    for name, digest in receipts.items():
        if file_sha(safe_path(output, name)) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    evidence = Evidence.model_validate_json(
        (output / "evidence.json").read_bytes()
    ).model_dump(mode="json", by_alias=True)
    manifest = json.loads((output / "manifest.json").read_bytes())
    ids = {r["image_id"] for r in manifest["images"]}
    selection = json.loads((output / "selection.json").read_bytes())
    selected = [r["image_id"] for r in selection["selected"]]
    case_ids = [c["image_id"] for c in evidence["cases"]]
    if len(set(case_ids)) != len(case_ids) or set(case_ids) != set(selected):
        raise ValueError("Incomplete or duplicated sample cases")
    if evidence["manifest_sha256"] != file_sha(output / "manifest.json") or evidence[
        "selection_sha256"
    ] != file_sha(output / "selection.json"):
        raise ValueError("Inventory/selection binding mismatch")
    with (output / "evidence.csv").open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    if [json.loads(r["case_json"]) for r in exported] != evidence["cases"]:
        raise ValueError("CSV/JSON mismatch")
    for row, case in zip(exported, evidence["cases"], strict=True):
        if (
            any(row[k] != case[k] for k in ("alignment_id", "image_id", "status"))
            or case["image_id"] not in ids
        ):
            raise ValueError("Case binding mismatch")
        if (
            case["evidence_locator"]
            and file_sha(safe_path(output, case["evidence_locator"]))
            != case["evidence_sha256"]
        ):
            raise ValueError("Visual evidence binding mismatch")
        if (
            case["control_evidence_locator"]
            and file_sha(safe_path(output, case["control_evidence_locator"]))
            != case["control_evidence_sha256"]
        ):
            raise ValueError("Control evidence binding mismatch")
    return evidence


def record_review(output, decision, destination):
    evidence = verify(output)
    review = HumanVerification.model_validate(decision)
    case = next(
        (c for c in evidence["cases"] if c["alignment_id"] == review.alignment_id), None
    )
    if (
        case is None
        or not case["evidence_locator"]
        or set(review.evidence_consulted) != {case["evidence_locator"]}
    ):
        raise ValueError("Human review must consult the registered case evidence")
    if review.source_locator != evidence["source_video"]:
        raise ValueError("Wrong source locator")
    if review.status == "human_verified" and review.selected_pts not in {
        m["pts"] for m in case["top_k"]
    }:
        raise ValueError("Selected PTS is not a documented candidate")
    destination = Path(destination)
    if destination.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("Reviews must be separate from the immutable evidence bundle")
    payload = encoded(
        {
            "review": review.model_dump(mode="json", by_alias=True),
            "evidence_document_sha256": file_sha(Path(output) / "evidence.json"),
            "sheet_sha256": case["evidence_sha256"],
        }
    )
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("Immutable review conflict; use a new revision file")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
    return {"recorded": True, "status": review.status}
