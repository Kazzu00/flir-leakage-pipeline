"""Offline regression fixtures reproduce both metadata revisions in the real TAR."""

import csv
import io
import json
import tarfile

import pytest
import yaml
from PIL import Image

from flir_pipeline.closeout.denoised import NAMES, prepare
from flir_pipeline.closeout.io import verify


def fixture_members(n=3):
    rows, members = [], []
    for i in range(n):
        name = f"video1_{i * 30 + 1:06d}_{i:04d}.000.png"
        img = io.BytesIO()
        Image.new("RGB", (8, 8), (i, 1, 2)).save(img, format="PNG")
        members.extend(
            [
                (f"images/{name}", img.getvalue()),
                (f"labels/{name[:-4]}.txt", b"0 0.5 0.5 0.2 0.2\n"),
            ]
        )
        rows.append(
            {
                "imagen": name,
                "video": "video1",
                "archivo_video": "source.mp4",
                "fps": "30",
                "frame": str(i * 30 + 1),
                "segundo": str(i),
                "nombre_original": f"original_{i}.jpg",
                "dataset_origen": "synthetic",
                "split_original": "train",
                "estado": "OK",
                "regimen": "none",
                "familia_denoising": "",
                "ruta_imagen_origen_en_este_dataset": f"reference/{name}",
                "n_cajas": "1",
                **{k: str(int(j == 0)) for j, k in enumerate(NAMES)},
            }
        )
    for class4 in ("SDZI", "maquinaria"):
        names = [*NAMES[:4], class4]
        members.append(
            (
                "clases.yaml",
                yaml.safe_dump({"nc": 5, "names": dict(enumerate(names))}).encode(),
            )
        )
        stream = io.StringIO(newline="")
        converted = [
            {(class4 if k == "maquinaria" else k): v for k, v in row.items()}
            for row in rows
        ]
        writer = csv.DictWriter(stream, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
        members.append(("correspondencia.csv", stream.getvalue().encode()))
    members += [
        (
            "resumen.json",
            json.dumps(
                {
                    "imagenes": n,
                    "etiquetas": n,
                    "faltantes": 0,
                    "por_familia": {"(ninguna)": n},
                    "ganadores": {},
                }
            ).encode(),
        ),
        ("faltantes.csv", b"imagen,familia,esperado\r\r\n"),
    ]
    return [("dataset_denoised/" + k, v) for k, v in members]


def make_tar(path, members=None):
    with tarfile.open(path, "w") as tar:
        for name, raw in members if members is not None else fixture_members():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return path


def test_regression_additional_metadata_and_versions(tmp_path):
    src = make_tar(tmp_path / "in.tar")
    result = prepare(src, tmp_path / "out", 3, True)
    assert result["deep_verified_images"] == result["images"] == 3
    assert result["metadata"]["missing_rows"] == []
    assert result["selected_versions"]["clases.yaml"] == 2
    verify(tmp_path / "out")
    with pytest.raises(FileExistsError):
        prepare(src, tmp_path / "out", 3)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "traversal",
        "absolute",
        "backslash",
        "unknown",
        "corrupt",
        "label",
        "csv",
        "class",
        "summary",
        "missing_conflict",
        "case_collision",
    ],
)
def test_reject_corruption_and_unsafe_members(tmp_path, mutation):
    members = fixture_members()
    if mutation == "missing":
        members.pop(1)
    elif mutation == "duplicate":
        members.append(members[0])
    elif mutation in ("traversal", "absolute", "backslash", "unknown"):
        name = {
            "traversal": "../escape.png",
            "absolute": "/escape",
            "backslash": "dataset_denoised\\images\\escape.png",
            "unknown": "dataset_denoised/unknown.json",
        }[mutation]
        members.append((name, b"bad"))
    elif mutation == "case_collision":
        members.append((members[0][0].upper(), members[0][1]))
    elif mutation in ("corrupt", "label"):
        index = int(mutation == "label")
        members[index] = (
            members[index][0],
            b"0 nan 0.5 0.2 0.2" if index else b"bad PNG",
        )
    else:
        suffix = {
            "csv": "correspondencia.csv",
            "class": "clases.yaml",
            "summary": "resumen.json",
            "missing_conflict": "faltantes.csv",
        }[mutation]
        index = max(i for i, (name, _) in enumerate(members) if name.endswith(suffix))
        raw = members[index][1]
        raw = (
            raw.replace(b"OK", b"NO")
            if mutation == "csv"
            else raw.replace(b"maquinaria", b"other")
            if mutation == "class"
            else raw.replace(b'"imagenes": 3', b'"imagenes": 2')
            if mutation == "summary"
            else raw + b"video1_000001_0000.000.png,family,x\n"
        )
        members[index] = (members[index][0], raw)
    with pytest.raises((ValueError, OSError)):
        prepare(make_tar(tmp_path / "in.tar", members), tmp_path / "out", 3, True)
    assert not (tmp_path / "out").exists()
    assert list(tmp_path.glob(".out.partial-*/FAILURE.json"))


def test_symlink_rejected(tmp_path):
    src = tmp_path / "link.tar"
    with tarfile.open(src, "w") as tar:
        info = tarfile.TarInfo("dataset_denoised/images/a.png")
        info.type, info.linkname = tarfile.SYMTYPE, "../../victim"
        tar.addfile(info)
    with pytest.raises(ValueError, match="Nonregular"):
        prepare(src, tmp_path / "out", 3)
