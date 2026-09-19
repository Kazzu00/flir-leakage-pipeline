"""Pinned MIT-licensed VIKUS runtime; downloaded only for an explicit local build.

No upstream JavaScript or images are vendored into the public repository. Small,
auditable presentation adaptations are applied to the generated local copy.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

from flir_pipeline.explorer.discovery import sha256

UPSTREAM = "https://github.com/cpietsch/vikus-viewer"
COMMIT = "ccb4e7e3d921521b01b2760b51bf4d1b090b844b"
ARCHIVE_SHA256 = "1295c3cb04e03317337ee29205438d7552c601d0834ac3b0640223c2a062e3f0"
ARCHIVE_URL = f"https://codeload.github.com/cpietsch/vikus-viewer/zip/{COMMIT}"
ADAPTER_VERSION = "flir-vikus-display-v1"


def fetch_runtime(cache: Path) -> Path:
    """Fetch fixed source bytes, without sending data or configuration upstream."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"vikus-{COMMIT}.zip"
    if not archive.exists():
        with urlopen(ARCHIVE_URL, timeout=60) as response:
            data = response.read(16 * 1024 * 1024)
        import hashlib

        if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Pinned VIKUS archive checksum differs")
        archive.write_bytes(data)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Cached VIKUS archive checksum differs")
    return archive


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("Pinned upstream patch context changed")
    return text.replace(old, new, 1)


def adapt_canvas(text: str) -> str:
    """Preserve ordering and original 2D geometry; never compute a new layout."""
    text = replace_once(text, "    left: 10,", "    left: 290,")
    text = replace_once(text, "var scale = 0.9 / Math.max(boxWidth / width, boxHeight / height);",
                        "var scale = 0.85 / Math.max(boxWidth / width, boxHeight / Math.max(200, height - 150));")
    text = replace_once(text, "      width / 2 - scale * (centerX + padding),",
                        "      margin.left + width / 2 - scale * (centerX + padding),")
    text = replace_once(text, "      height / 2 - scale * (height + centerY + padding),",
                        "      (height + 150) / 2 - scale * (height + centerY + padding),")
    text = replace_once(text, "    timelineData = _timeline;", "    timelineData = _timeline;\n    state.mode = config.loader.layouts[0];")
    text = replace_once(text, "        return b.keywords.length - a.keywords.length;",
                        "        return Number(a._display_order) - Number(b._display_order);")
    text = replace_once(text, "        return a - b;",
                        "        var order = config.sortArrays[groupKey] || [];\n        return order.indexOf(a) - order.indexOf(b);")
    text = replace_once(text,
                        "    var x = d3.scale.linear().range([0, 1]).domain(xExtent);\n    var y = d3.scale.linear().range([0, 1]).domain(yExtent);",
                        "    // FLIR: one common scale preserves the saved geometry's aspect ratio.\n"
                        "    var span = Math.max(xExtent[1]-xExtent[0], yExtent[1]-yExtent[0]) || 1;\n"
                        "    var cx = (xExtent[0]+xExtent[1])/2, cy = (yExtent[0]+yExtent[1])/2;\n"
                        "    var x = d3.scale.linear().range([0, 1]).domain([cx-span/2, cx+span/2]);\n"
                        "    var y = d3.scale.linear().range([0, 1]).domain([cy-span/2, cy+span/2]);")
    # Upstream moves filtered-out items into a random ring. Keep every item at
    # its original reduced position instead; filtering changes opacity only.
    start = text.index("    var marginBottom = -height / 2.5;", text.index("canvas.projectTSNE ="))
    end = text.index("      var factor = height / 2;", start)
    text = text[:start] + "    var dimension = Math.min(width, height) * 0.8;\n    data.forEach(function (d) {\n" + text[end:]
    return text


def install_runtime(archive: Path, destination: Path) -> dict:
    """Copy pinned static assets and license to a new, local generated bundle."""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Runtime archive is not the pinned upstream")
    prefix = f"vikus-viewer-{COMMIT}/"
    copied = {}
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as source:
        for member in source.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix):]
            if relative not in {"index.html", "LICENSE.md", "README.md"} and not relative.startswith(("js/", "css/", "font/", "img/")):
                continue
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe upstream archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            data = source.read(member)
            if relative == "index.html":
                text = data.decode("utf-8")
                text = re.sub(r"<!--\[if lt IE 9\]>.*?<!\[endif\]-->", "", text, flags=re.S)
                text = replace_once(text, '<link rel="stylesheet" href="css/timeline.css" />',
                                    '<link rel="stylesheet" href="css/timeline.css" />\n  <link rel="stylesheet" href="flir.css" />')
                text = replace_once(text, '<script src="js/viz.js"></script>',
                                    '<script src="flir.js"></script>\n  <script src="js/viz.js"></script>')
                text = replace_once(text, "<body>", '<body>\n  <header id="flir-header"><strong>FLIR · Scene / Cluster</strong><span id="flir-experiment"></span><span id="flir-status">Cargando colección local…</span><button id="flir-focus">Enfocar selección</button><button id="flir-reset">Restablecer filtros</button></header>')
                data = text.encode("utf-8")
            elif relative == "js/canvas.js":
                data = adapt_canvas(data.decode("utf-8")).encode("utf-8")
            path.write_bytes(data)
            copied[relative] = sha256(path)
    for name in ("flir.js", "flir.css"):
        destination.joinpath(name).write_bytes(Path(__file__).with_name("vikus_assets").joinpath(name).read_bytes())
    return {"repository": UPSTREAM, "commit": COMMIT, "license": "MIT",
            "archive_sha256": ARCHIVE_SHA256, "adapter_version": ADAPTER_VERSION,
            "adapted_files": ["index.html", "js/canvas.js"], "runtime_sha256": copied}
