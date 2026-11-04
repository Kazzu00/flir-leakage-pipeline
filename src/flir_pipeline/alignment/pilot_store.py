"""Append-only alignment judgments, separate from all historical audit ledgers."""

import csv
import io
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .inventory import encoded, file_sha, safe_path
from .pilot import load_pilot

Label = Literal["correct_match", "incorrect_match", "uncertain"]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alignment_id: str
    expected_version: int = Field(ge=0)
    status: Literal["pending", "reviewed"] = "pending"
    label: Label | None = None
    confidence: Literal["high", "medium", "low", "undetermined"] = "undetermined"
    justification: str = Field(default="", max_length=10000)
    reviewer: str = Field(default="", max_length=200)
    evidence_consulted: list[str] = Field(default_factory=list)
    better_rank: int | None = Field(default=None, ge=2, le=10)
    no_correspondence_in_presented_evidence: bool = False
    control_label: Label | None = None

    @model_validator(mode="after")
    def justified(self):
        if self.status == "reviewed":
            if (
                self.label is None
                or not self.reviewer.strip()
                or not self.justification.strip()
                or not self.evidence_consulted
            ):
                raise ValueError(
                    "Reviewed alignment requires label, reviewer, justification and consulted evidence"
                )
            if (
                self.label != "uncertain"
                or self.control_label in {"correct_match", "incorrect_match"}
            ) and self.confidence == "undetermined":
                raise ValueError(
                    "Conclusive alignment judgment requires explicit human confidence"
                )
        if self.label == "correct_match" and (
            self.better_rank or self.no_correspondence_in_presented_evidence
        ):
            raise ValueError(
                "Top-1 correct conflicts with better alternative/no correspondence"
            )
        if self.better_rank and self.no_correspondence_in_presented_evidence:
            raise ValueError("Better alternative conflicts with no correspondence")
        return self


class PilotStore:
    def __init__(self, pilot, db):
        self.root, self.path = Path(pilot).resolve(), Path(db).resolve()
        self.data = load_pilot(self.root)
        self.binding = file_sha(self.root / "pilot.json")
        if self.path.is_relative_to(self.root):
            raise ValueError("Ledger must be outside immutable pilot bundle")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with sqlite3.connect(self.path) as connection:
                tables = {
                    r[0]
                    for r in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if "pilot_meta" not in tables:
                    raise ValueError("Refusing to modify a foreign/historical database")
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pilot_meta (binding TEXT PRIMARY KEY)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, alignment_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, UNIQUE(alignment_id,version))"
            )
            old = connection.execute("SELECT binding FROM pilot_meta").fetchone()
            if old and old[0] != self.binding:
                raise ValueError("Pilot binding conflict")
            connection.execute(
                "INSERT OR IGNORE INTO pilot_meta VALUES (?)", (self.binding,)
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def snapshot(self):
        if file_sha(self.root / "pilot.json") != self.binding:
            raise ValueError("Pilot document changed")
        with self.connect() as connection:
            history = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT payload FROM reviews ORDER BY rowid"
                )
            ]
        return {"pilot": self.data, "history": history, "pilot_sha256": self.binding}

    def save(self, value):
        decision = Decision.model_validate(value)
        case = next(
            (
                c
                for c in self.data["cases"]
                if c["alignment_id"] == decision.alignment_id
            ),
            None,
        )
        if case is None:
            raise ValueError("Unknown case")
        allowed = {case["original_media"]}
        for alternative in case["alternatives"]:
            allowed.update(
                filter(
                    None,
                    [
                        alternative["frame_media"],
                        alternative["lines_media"],
                        alternative["overlay_media"],
                    ],
                )
            )
        if case["control"]:
            allowed.add(case["control"]["media"])
        consulted = set(decision.evidence_consulted)
        if not consulted <= allowed:
            raise ValueError("Consulted evidence does not belong to this case")
        required = {case["original_media"]}
        if case["alternatives"]:
            required.add(case["alternatives"][0]["frame_media"])
        elif decision.status == "reviewed" and decision.label == "correct_match":
            raise ValueError("No top-1 candidate to confirm")
        if decision.better_rank:
            alternative = next(
                (a for a in case["alternatives"] if a["rank"] == decision.better_rank),
                None,
            )
            if alternative is None:
                raise ValueError("Unknown alternative rank")
            required.add(alternative["frame_media"])
        if decision.control_label:
            if case["control"] is None:
                raise ValueError("No registered control")
            required.add(case["control"]["media"])
        if decision.status == "reviewed" and not required <= consulted:
            raise ValueError(
                "Consult original, top-1 and any judged alternative/control explicitly"
            )
        if file_sha(self.root / "pilot.json") != self.binding:
            raise ValueError("Pilot document changed")
        for key in consulted:
            self.media(key)  # Check immutable bytes at decision time.
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT coalesce(max(version),0) FROM reviews WHERE alignment_id=?",
                (decision.alignment_id,),
            ).fetchone()[0]
            if current != decision.expected_version:
                raise ValueError("Version conflict: reload before saving")
            payload = {
                **decision.model_dump(),
                "version": current + 1,
                "decision_id": str(uuid.uuid4()),
                "created_at": datetime.now(UTC).isoformat(),
                "pilot_sha256": self.binding,
                "evidence_hashes": {
                    k: self.data["media"][k]["sha256"] for k in sorted(consulted)
                },
                "human_verified_top1": decision.status == "reviewed"
                and decision.label == "correct_match",
                "schema": "alignment-pilot-review-v1",
            }
            connection.execute(
                "INSERT INTO reviews VALUES (?,?,?,?)",
                (
                    payload["decision_id"],
                    decision.alignment_id,
                    current + 1,
                    encoded(payload).decode(),
                ),
            )
        return payload

    def media(self, key):
        item = self.data["media"].get(key)
        if item is None:
            raise ValueError("Unknown media ID")
        path = safe_path(self.root, item["path"])
        if file_sha(path) != item["sha256"]:
            raise ValueError("Media checksum mismatch")
        return path.read_bytes()

    def export(self, fmt="json"):
        snapshot = self.snapshot()
        latest = {d["alignment_id"]: d for d in snapshot["history"]}
        initial = [
            {
                "alignment_id": c["alignment_id"],
                "version": 0,
                "status": "pending",
                "label": None,
                "pilot_sha256": self.binding,
                "origin": "implicit_unreviewed_case_not_a_human_decision",
            }
            for c in self.data["cases"]
            if c["alignment_id"] not in latest
        ]
        if fmt == "json":
            return encoded(
                {
                    "schema": "alignment-pilot-export-v1",
                    "pilot_sha256": self.binding,
                    "cases": [c["alignment_id"] for c in self.data["cases"]],
                    "history": snapshot["history"],
                    "initial_pending": initial,
                }
            )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "record_type",
                "alignment_id",
                "version",
                "status",
                "label",
                "decision_json",
            ],
        )
        writer.writeheader()
        for decision in snapshot["history"] + initial:
            writer.writerow(
                {
                    "record_type": "history"
                    if decision["version"]
                    else "initial_pending",
                    **{
                        k: decision[k]
                        for k in ("alignment_id", "version", "status", "label")
                    },
                    "decision_json": json.dumps(decision, sort_keys=True),
                }
            )
        return stream.getvalue().encode("utf-8")

    def report(self, minimum=5):
        if minimum < 1:
            raise ValueError("Minimum must be positive")
        history = self.snapshot()["history"]
        latest = {d["alignment_id"]: d for d in history}
        cases = self.data["cases"]

        def counts(subset):
            reviewed = [
                latest[c["alignment_id"]]
                for c in subset
                if c["alignment_id"] in latest
                and latest[c["alignment_id"]]["status"] == "reviewed"
            ]
            labels = {
                k: sum(d["label"] == k for d in reviewed)
                for k in ("correct_match", "incorrect_match", "uncertain")
            }
            conclusive = labels["correct_match"] + labels["incorrect_match"]
            return {
                "cases": len(subset),
                "reviewed": len(reviewed),
                "pending": len(subset) - len(reviewed),
                **labels,
                "correct_among_reviewed": labels["correct_match"] / len(reviewed)
                if len(reviewed) >= minimum
                else None,
                "correct_among_conclusive": labels["correct_match"] / conclusive
                if conclusive >= minimum
                else None,
                "better_than_top1": sum(d["better_rank"] is not None for d in reviewed),
                "no_correspondence_presented": sum(
                    d["no_correspondence_in_presented_evidence"] for d in reviewed
                ),
            }

        controls = [
            d
            for d in latest.values()
            if d["status"] == "reviewed" and d["control_label"] is not None
        ]
        abstentions = [c for c in cases if c["automatic_status"] == "not_found"]
        references = [
            latest[c["alignment_id"]]
            for c in abstentions
            if c["alignment_id"] in latest
            and latest[c["alignment_id"]]["status"] == "reviewed"
        ]
        reference_table = {
            "abstained_no_correspondence": 0,
            "abstained_correct_top1": 0,
            "proposed_no_correspondence": 0,
            "proposed_correct_top1": 0,
        }
        for case in cases:
            decision = latest.get(case["alignment_id"])
            if not decision or decision["status"] != "reviewed":
                continue
            prefix = (
                "abstained" if case["automatic_status"] == "not_found" else "proposed"
            )
            if decision["no_correspondence_in_presented_evidence"]:
                reference_table[prefix + "_no_correspondence"] += 1
            elif decision["label"] == "correct_match":
                reference_table[prefix + "_correct_top1"] += 1
        negatives = (
            reference_table["abstained_no_correspondence"]
            + reference_table["proposed_no_correspondence"]
        )
        return {
            "schema": "alignment-pilot-analysis-v1",
            "minimum_reviewed_per_group": minimum,
            "overall": counts(cases),
            "by_split": {
                s: counts([c for c in cases if c["original"]["original_split"] == s])
                for s in ("train", "val", "test")
            },
            "by_automatic_confidence": {"not_available_not_calibrated": counts(cases)},
            "by_support_band": {
                band: counts([c for c in cases if c["support_band"] == band])
                for band in sorted({c["support_band"] for c in cases})
            },
            "controls": {
                "reviewed": len(controls),
                "human_rejected": sum(
                    d["control_label"] == "incorrect_match" for d in controls
                ),
                "human_accepted": sum(
                    d["control_label"] == "correct_match" for d in controls
                ),
                "uncertain": sum(d["control_label"] == "uncertain" for d in controls),
            },
            "abstention": {
                "reference_table": reference_table,
                "fraction_abstained_among_human_no_correspondence": reference_table[
                    "abstained_no_correspondence"
                ]
                / negatives
                if negatives >= minimum
                else None,
                "automatic_count": len(abstentions),
                "human_reviewed": len(references),
                "human_no_correspondence": sum(
                    d["no_correspondence_in_presented_evidence"] for d in references
                ),
                "missed_correct_top1": sum(
                    d["label"] == "correct_match" for d in references
                ),
            },
            "limitation": "Directed pilot, descriptive human references only. No calibrated confidence, population accuracy or geographic independence claims.",
        }
