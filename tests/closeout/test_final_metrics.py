"""Synthetic invariants for lossless publication, without private FLIR inputs."""

import csv
import hashlib
import io

import pytest

from flir_pipeline.closeout.final_metrics import FINAL_COLUMNS, project


def fixture(rows=None):
    if rows is None:
        rows = [
            [t, p, c, "0.123456789012345678901234567890", "0.20", "1e-3", "0"]
            for t in ("AUG_ON", "AUG_OFF")
            for p in ("historica", "aleatoria", "c10")
            for c in ("five_class_macro", "shared_01_macro")
        ]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["treatment", "policy", "cohort", "precision", "recall", "map50", "map50_95"])
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    return raw, {
        "state": "AGGREGATED",
        "protocol": "oe4_closeout_v2",
        "files": {"comparison.csv": hashlib.sha256(raw).hexdigest()},
    }


def test_exact_tokens_order_and_reverse_projection():
    source, receipt = fixture()
    result = project(source, receipt)
    reader = csv.DictReader(io.StringIO(result.decode()))
    assert reader.fieldnames == list(FINAL_COLUMNS)
    rows = list(reader)
    assert [(r["policy"], r["augmentation"]) for r in rows] == [
        (p, t) for p in ("historica", "aleatoria", "c10") for t in ("AUG_ON", "AUG_OFF")
    ]
    restored = {}
    for row in rows:
        for cohort, suffix in [("five_class_macro", "5class"), ("shared_01_macro", "shared_01")]:
            restored[row["policy"], row["augmentation"], cohort] = [
                row[f"{m}_{suffix}"] for m in ("precision", "recall", "map50", "map50_95")
            ]
    for row in csv.DictReader(io.StringIO(source.decode())):
        assert restored[row["policy"], row["treatment"], row["cohort"]] == [
            row[m] for m in ("precision", "recall", "map50", "map50_95")
        ]


def test_tampered_source_rejected():
    source, receipt = fixture()
    with pytest.raises(ValueError, match="SHA-256"):
        project(source + b"\n", receipt)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unexpected", "nan", "range"])
def test_invalid_cells_rejected(mutation):
    source, _ = fixture()
    rows = list(csv.reader(io.StringIO(source.decode())))[1:]
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unexpected":
        rows[0][2] = "unknown_cohort"
    elif mutation == "nan":
        rows[0][3] = "NaN"
    else:
        rows[0][3] = "1.001"
    source, receipt = fixture(rows)
    with pytest.raises(ValueError):
        project(source, receipt)
