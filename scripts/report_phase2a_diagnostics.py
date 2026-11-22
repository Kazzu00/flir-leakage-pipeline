"""Build local four-panel boundary evidence from a verified alignment bundle.

Index pairs select diagnostics only. They do not label continuity or region identity.
Outputs stay separate from the immutable matching run.
"""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from flir_pipeline.alignment.inventory import encoded, file_sha, read_image, safe_path
from flir_pipeline.alignment.matching import Decoder
from flir_pipeline.alignment.pipeline import verify


def build(root, evidence_dir, output):
    root, evidence_dir, output = map(
        lambda p: Path(p).resolve(), (root, evidence_dir, output)
    )
    if output.is_relative_to(root) or output.is_relative_to(evidence_dir):
        raise ValueError(
            "Diagnostic output must be separate from originals and evidence"
        )
    if output.exists():
        raise ValueError("Use a new diagnostic output directory")
    evidence = verify(evidence_dir)
    manifest = json.loads((evidence_dir / "manifest.json").read_bytes())
    rows = {r["image_id"]: r for r in manifest["images"]}
    cases = {}
    for case in evidence["cases"]:
        cases.setdefault(rows[case["image_id"]]["dataset_index"], []).append(case)
    video = safe_path(root, evidence["source_video"])
    if file_sha(video) != evidence["source_video_sha256"]:
        raise ValueError("Source video changed")
    output.mkdir(parents=True)
    decoder = Decoder(video, max_frames=3000, max_seconds=300)
    records = []
    try:
        for left, right in [(199, 200), (349, 350), (399, 400), (499, 501)]:
            record = {
                "indices": [left, right],
                "verification": "candidate_only",
                "cases": [],
                "missing_indices": [],
                "ambiguous_occurrence_indices": [],
                "sheet": None,
            }
            panels = []
            for index in (left, right):
                options = cases.get(index, [])
                if len(options) > 1:
                    record["ambiguous_occurrence_indices"].append(index)
                if len(options) != 1 or not options[0]["top_k"]:
                    record["missing_indices"].append(index)
                    continue
                case = options[0]
                row = rows[case["image_id"]]
                _, original = read_image(root, row)
                match = case["top_k"][0]
                frames = decoder.frames(
                    match["second"] - 1e-8, match["second"] + 0.1, 0
                )
                found = next((f for f in frames if f["pts"] == match["pts"]), None)
                if (
                    found is None
                    or found["decoded_rgb_sha256"] != match["decoded_rgb_sha256"]
                ):
                    raise ValueError(
                        "Independent re-extraction differs from recorded candidate"
                    )
                panels.extend(
                    [
                        (f"Original index {index}", original),
                        (
                            f"Candidate PTS {match['pts']} / {match['second']:.6f} s",
                            Image.fromarray(found["rgb"]),
                        ),
                    ]
                )
                record["cases"].append(
                    {
                        "alignment_id": case["alignment_id"],
                        "image_id": case["image_id"],
                        "original_split": row["original_split"],
                        "status": case["status"],
                        "pts": match["pts"],
                        "second": match["second"],
                        "time_base": match["time_base"],
                    }
                )
            if panels:
                canvas = Image.new(
                    "RGB", (768, 250 * ((len(panels) + 1) // 2)), "white"
                )
                draw = ImageDraw.Draw(canvas)
                for i, (label, image) in enumerate(panels):
                    x, y = (i % 2) * 384, (i // 2) * 250
                    canvas.paste(image.resize((384, 216)), (x, y + 24))
                    draw.text((x + 4, y + 4), label, fill="black")
                name = f"boundary-{left}-{right}.png"
                canvas.save(output / name)
                record.update(sheet=name, sheet_sha256=file_sha(output / name))
            records.append(record)
    finally:
        decoder.close()
    payload = {
        "schema": "alignment-diagnostic-view-v1",
        "evidence_sha256": file_sha(evidence_dir / "evidence.json"),
        "report_script_sha256": file_sha(__file__),
        "decoded_frames": decoder.decoded,
        "pairs": records,
        "decode_log": decoder.seek_log,
        "warning": "No continuity, geographic identity or independence judgments are made.",
    }
    (output / "diagnostics.json").write_bytes(encoded(payload))
    cards = []
    for row in records:
        image = (
            f'<img src="{row["sheet"]}" alt="Original images and their individual source candidates">'
            if row["sheet"]
            else "Missing evidence"
        )
        cards.append(
            f"<section><h2>{row['indices']}</h2>{image}<details><summary>Provenance and candidate states</summary><pre>{html.escape(json.dumps(row, indent=2))}</pre></details></section>"
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>FLIR boundary diagnostics</title><style>body{font:16px sans-serif;max-width:900px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Boundary diagnostics: human assessment pending</h1><p>These are individual image-to-video candidates. Inspect direct evidence before assessing continuity and region dependence independently. Index 500 is not substituted.</p>'
        + "".join(cards),
        encoding="utf-8",
    )
    return {
        "pairs": len(records),
        "complete_pairs": sum(not r["missing_indices"] for r in records),
        "decoded_frames": decoder.decoded,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.evidence, args.output), indent=2))
