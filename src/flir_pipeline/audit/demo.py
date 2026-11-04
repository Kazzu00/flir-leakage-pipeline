"""Explicitly SYNTHETIC local smoke fixture; no FAC inputs or model downloads."""

import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from PIL import Image, ImageDraw


def create_demo(root: Path):
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    with ZipFile(root / "synthetic_review.zip", "w") as archive:
        for i in range(3):
            image = Image.new("RGB", (640, 240), "#252535")
            draw = ImageDraw.Draw(image)
            draw.text((15, 12), "SYNTHETIC - Before / After - NOT FAC", fill="white")
            draw.rectangle((40, 65, 230, 210), fill="#8e9bba")
            draw.rectangle((360, 65, 550, 210), fill="#8e9bba" if i != 2 else "#d5a46a")
            stream = io.BytesIO()
            image.save(stream, "PNG")
            name = f"synthetic_{i}.png"
            archive.writestr(ZipInfo(name, (1980, 1, 1, 0, 0, 0)), stream.getvalue())
            rows.append(
                {
                    "review_id": f"SYNTHETIC-{i}",
                    "approx_index": i,
                    "approx_source_second": float(i),
                    "image_file": name,
                    "kind": "synthetic_unclassified",
                }
            )
        archive.writestr(
            ZipInfo("manifest.json", (1980, 1, 1, 0, 0, 0)),
            json.dumps(
                {
                    "source_video": "SYNTHETIC-NO-VIDEO",
                    "synthetic": True,
                    "results": rows,
                }
            ),
        )
