"""Independent reviewer ledgers; append-only judgments and explicit adjudication."""

import csv
import io
import json
import sqlite3
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dependency import load
from .inventory import encoded, file_sha, safe_path


class PairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str = Field(min_length=1)
    expected_version: int = Field(ge=0)
    status: Literal["pending", "reviewed"] = "pending"
    continuity: Literal["continuous", "shot_change", "uncertain"] = "uncertain"
    spatial: Literal["same_region", "different_region", "uncertain"] = "uncertain"
    confidence: Literal["high", "medium", "low", "undetermined"] = "undetermined"
    justification: str = Field(default="", max_length=10000)
    reviewer: str = Field(default="", max_length=200)
    phenomena: list[
        Literal[
            "zoom",
            "rapid_camera",
            "signal_loss",
            "gain",
            "polarity",
            "occlusion",
            "low_texture",
        ]
    ] = Field(default_factory=list)
    evidence_consulted: list[str] = Field(default_factory=list)
    adjudicates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def justified(self):
        if self.status == "reviewed":
            if (
                not self.reviewer.strip()
                or not self.justification.strip()
                or not self.evidence_consulted
            ):
                raise ValueError(
                    "Review requires reviewer, justification and explicit consulted evidence"
                )
            if (
                self.continuity != "uncertain" or self.spatial != "uncertain"
            ) and self.confidence == "undetermined":
                raise ValueError("Conclusive judgment requires confidence")
        if len(set(self.evidence_consulted)) != len(self.evidence_consulted):
            raise ValueError("Duplicate consulted evidence")
        return self


class StoredDecision(PairDecision):
    id: str
    version: int = Field(ge=1)
    created_at: str
    role: Literal["reviewer1", "reviewer2", "adjudicator"]
    pilot_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    evidence_hashes: dict[str, str]


class ReviewExport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["dependency-review-export-v1"] = Field(alias="schema")
    pilot_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    role: Literal["reviewer1", "reviewer2", "adjudicator"]
    history: list[StoredDecision]
    initial_pending: list[str]


def read_ledger(path, binding, role):
    """Read independent annotations without creating or mutating a database."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        if db.execute("SELECT binding, role FROM dependency_meta").fetchone() != (
            binding,
            role,
        ):
            raise ValueError("Reviewer ledger binding/role mismatch")
        return [
            json.loads(r[0])
            for r in db.execute("SELECT payload FROM dependency_reviews ORDER BY rowid")
        ]


class DependencyStore:
    def __init__(self, pilot, db, role="reviewer1", sources=()):
        if role not in {"reviewer1", "reviewer2", "adjudicator"}:
            raise ValueError("Unknown role")
        self.root, self.path = Path(pilot).resolve(), Path(db).resolve()
        self.data, self.role = load(self.root), role
        self.binding = file_sha(self.root / "pilot.json")
        self.source_rows = []
        if role == "adjudicator":
            if len(sources) != 2:
                raise ValueError("Adjudication requires both independent ledgers")
            for path, slot in zip(sources, ("reviewer1", "reviewer2"), strict=True):
                self.source_rows.extend(read_ledger(path, self.binding, slot))
        elif sources:
            raise ValueError("Blind reviewer cannot load another reviewer's ledger")
        if self.path.is_relative_to(self.root):
            raise ValueError("Ledger must be outside immutable evidence")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            read_ledger(
                self.path, self.binding, role
            )  # Refuse foreign DB before writes.
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS dependency_meta(binding TEXT, role TEXT)"
            )
            if not db.execute("SELECT 1 FROM dependency_meta").fetchone():
                db.execute(
                    "INSERT INTO dependency_meta VALUES (?,?)", (self.binding, role)
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS dependency_reviews(id TEXT PRIMARY KEY, pair_id TEXT, version INTEGER, payload TEXT, UNIQUE(pair_id,version))"
            )
        self.cases = {
            p["pair_id"]: p
            for p in self.data["pairs"]
            if role != "reviewer2" or p["pair_id"] in self.data["second_reviewer_pairs"]
        }

    def history(self):
        return read_ledger(self.path, self.binding, self.role)

    def media(self, key):
        if not any(key in p["media"] for p in self.cases.values()):
            raise ValueError("Media not assigned to this reviewer")
        item = self.data["media"][key]
        path = safe_path(self.root, item["path"])
        if file_sha(path) != item["sha256"]:
            raise ValueError("Evidence changed")
        return path.read_bytes()

    def snapshot(self):
        # No cross-reviewer endpoint: independent ledger and assignment enforced server-side.
        return {
            "pilot": {
                "schema": self.data["schema"],
                "pairs": list(self.cases.values()),
                "media": {
                    k: v
                    for k, v in self.data["media"].items()
                    if any(k in p["media"] for p in self.cases.values())
                },
                "method": self.data["method"],
                "source_video": self.data["source_video"],
            },
            "role": self.role,
            "binding": self.binding,
            "history": self.history(),
            "adjudication_sources": self.source_rows
            if self.role == "adjudicator"
            else [],
        }

    def save(self, value):
        decision = PairDecision.model_validate(value)
        if decision.pair_id not in self.cases:
            raise ValueError("Pair not assigned to reviewer")
        case = self.cases[decision.pair_id]
        if not set(decision.evidence_consulted) <= set(case["media"]):
            raise ValueError("Evidence belongs to another pair")
        if decision.status == "reviewed":
            required = {case[s]["media"][0] for s in ("a", "b")}
            if not required <= set(decision.evidence_consulted):
                raise ValueError("Both original images must be explicitly consulted")
        if self.role == "adjudicator":
            refs = {
                r["id"]: r
                for r in self.source_rows
                if r["pair_id"] == decision.pair_id and r["status"] == "reviewed"
            }
            selected = [refs.get(k) for k in decision.adjudicates]
            if (
                len(selected) != 2
                or any(r is None for r in selected)
                or {r["role"] for r in selected} != {"reviewer1", "reviewer2"}
            ):
                raise ValueError(
                    "Adjudication must reference two real independent decisions"
                )
        elif decision.adjudicates:
            raise ValueError("Only adjudicator may adjudicate")
        for key in decision.evidence_consulted:
            self.media(key)
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute(
                "SELECT COALESCE(MAX(version),0) FROM dependency_reviews WHERE pair_id=?",
                (decision.pair_id,),
            ).fetchone()[0]
            if version != decision.expected_version:
                raise ValueError("Version conflict: reload current decision")
            row = dict(
                decision.model_dump(),
                id=str(uuid.uuid4()),
                version=version + 1,
                created_at=datetime.now(UTC).isoformat(),
                role=self.role,
                pilot_sha256=self.binding,
                evidence_hashes={
                    k: self.data["media"][k]["sha256"]
                    for k in decision.evidence_consulted
                },
            )
            db.execute(
                "INSERT INTO dependency_reviews VALUES (?,?,?,?)",
                (row["id"], decision.pair_id, version + 1, encoded(row).decode()),
            )
        return row

    def export(self, fmt):
        history = self.history()
        pending = [p for p in self.cases if p not in {r["pair_id"] for r in history}]
        if fmt == "json":
            return encoded(
                {
                    "schema": "dependency-review-export-v1",
                    "pilot_sha256": self.binding,
                    "role": self.role,
                    "history": history,
                    "initial_pending": pending,
                }
            )
        if fmt != "csv":
            raise ValueError("Use json or csv")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["record_type", "pair_id", "version", "decision_json"])
        for row in history:
            writer.writerow(
                [
                    "history",
                    row["pair_id"],
                    row["version"],
                    json.dumps(row, ensure_ascii=False),
                ]
            )
        for pair in pending:
            writer.writerow(["initial_pending", pair, 0, ""])
        return output.getvalue().encode("utf-8")

    def report(self):
        return summarize(self.data, self.history())


def summarize(data, history):
    latest = {(r["role"], r["pair_id"]): r for r in history}
    reviewed = [
        r
        for r in latest.values()
        if r["status"] == "reviewed" and r["role"] != "adjudicator"
    ]
    pairs = {p["pair_id"]: p for p in data["pairs"]}
    comparisons = []
    for pid in pairs:
        a, b = (latest.get((role, pid)) for role in ("reviewer1", "reviewer2"))
        if a and b and a["status"] == b["status"] == "reviewed":
            comparisons.append(
                {
                    "pair_id": pid,
                    "continuity_agrees": a["continuity"] == b["continuity"],
                    "spatial_agrees": a["spatial"] == b["spatial"],
                    "decision_ids": [a["id"], b["id"]],
                }
            )
    return {
        "schema": "dependency-report-v1",
        "pair_count": len(pairs),
        "types": dict(Counter(p["category"] for p in pairs.values())),
        "reviewed_decisions": len(reviewed),
        "pending_first_reviewer": sum(
            latest.get(("reviewer1", pid), {}).get("status") != "reviewed"
            for pid in pairs
        ),
        "by_role": {
            role: {
                axis: dict(Counter(r[axis] for r in reviewed if r["role"] == role))
                for axis in ("continuity", "spatial")
            }
            for role in ("reviewer1", "reviewer2")
        },
        "cross_split_reviewed": [
            r for r in reviewed if pairs[r["pair_id"]]["secondary"]["cross_split"]
        ],
        "agreement": {
            "paired_count": len(comparisons),
            "pairs": comparisons,
            "continuity_fraction": sum(r["continuity_agrees"] for r in comparisons)
            / len(comparisons),
            "spatial_fraction": sum(r["spatial_agrees"] for r in comparisons)
            / len(comparisons),
        }
        if comparisons
        else None,
        "adjudications": [r for r in latest.values() if r["role"] == "adjudicator"],
        "limitations": [
            "Directed pilot, not population leakage rates",
            "same_region does not prove contamination or detector impact",
            METHOD_LIMIT,
        ],
    }


METHOD_LIMIT = "Sparse context and uncertain alignment times do not establish independent sequences; negative controls are unverified candidates"
