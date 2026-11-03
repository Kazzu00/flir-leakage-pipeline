"""Transactional local ledger. Raw inputs and append-only decisions share one DB.

Each artifact revision has its own identity. A changed input never silently
inherits a judgment. SQLite commits make interruption rollback atomic; optimistic
review versions prevent two browser tabs overwriting one another.
"""

import csv
import io
import json
import os
import sqlite3
import tempfile
import uuid
import zlib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .importers import (
    MAX_FILE,
    PRIORITY,
    compare_records,
    decode,
    digest,
    encode,
    normalize,
)
from .models import SCHEMA, CandidateEvent, DecisionInput, ReviewDecision


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (schema TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, sha TEXT NOT NULL, origin TEXT NOT NULL,
                    raw BLOB NOT NULL, normalized TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, asset_id TEXT REFERENCES assets(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY, asset_id TEXT REFERENCES assets(id), payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY, event_id TEXT REFERENCES events(id),
                    version INTEGER NOT NULL, payload TEXT NOT NULL, UNIQUE(event_id,version));
                CREATE TABLE IF NOT EXISTS imports (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, report TEXT NOT NULL);
            """)
            rows = db.execute("SELECT schema FROM metadata").fetchall()
            if not rows:
                db.execute("INSERT INTO metadata VALUES (?)", (SCHEMA,))
            elif rows != [(SCHEMA,)]:
                raise ValueError("Unsupported database schema")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def import_directory(self, root: Path, priority_only=False):
        root = root.resolve(strict=True)
        if self.path.resolve().is_relative_to(root):
            raise ValueError("Database must be outside the original input directory")
        report = {
            "schema": SCHEMA,
            "files": [],
            "missing": [n for n in PRIORITY if not (root / n).is_file()],
        }
        paths = sorted(
            p for p in root.iterdir() if p.suffix.lower() in {".json", ".csv", ".zip"}
        )
        report["excluded_by_priority_selection"] = [
            p.name for p in paths if priority_only and p.name not in PRIORITY
        ]
        if priority_only:
            paths = [p for p in paths if p.name in PRIORITY]
        # Unsupported auxiliary files are reported; nothing is fabricated for them.
        for path in paths:
            entry = {"file": path.name}
            try:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size > MAX_FILE
                ):
                    raise ValueError("Not a regular bounded artifact (limit 64 MiB)")
                raw = path.read_bytes()
                entry["sha256"] = digest(raw)
                asset, events, observations, media, warnings = normalize(raw, path.name)
                alternate = (
                    path.suffix == ".csv" and path.with_suffix(".json").is_file()
                )
                if alternate:
                    companion = path.with_suffix(".json")
                    if companion.stat().st_size > MAX_FILE:
                        raise ValueError("Oversized JSON companion")
                    # Validate the primary too; a corrupt JSON cannot hide valid CSV silently.
                    normalize(companion.read_bytes(), companion.name)
                    compare_records(
                        decode(companion.read_bytes(), ".json"), decode(raw, ".csv")
                    )
                    events, observations = [], []
                    warnings.append(
                        "CSV retained as companion; JSON supplies normalized records."
                    )
                frozen = {
                    "asset": asset,
                    "events": events,
                    "observations": observations,
                    "media": media,
                    "csv_companion": alternate,
                }
                with self.connect() as db:
                    db.execute(
                        "INSERT OR IGNORE INTO assets VALUES (?,?,?,?,?)",
                        (
                            asset["asset_id"],
                            asset["sha256"],
                            str(path.resolve()),
                            raw,
                            encode(frozen),
                        ),
                    )
                    for table, items, key in [
                        ("events", events, "event_id"),
                        ("observations", observations, "observation_id"),
                    ]:
                        db.executemany(
                            f"INSERT OR IGNORE INTO {table} VALUES (?,?,?)",
                            [(r[key], asset["asset_id"], encode(r)) for r in items],
                        )
                entry.update(
                    status="valid",
                    events=len(events),
                    observations=len(observations),
                    warnings=warnings,
                )
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                RuntimeError,
                BadZipFile,
                csv.Error,
                zlib.error,
                OverflowError,
            ) as exc:
                entry.update(status="error", error=str(exc))
            report["files"].append(entry)
        with self.connect() as db:
            db.execute(
                "INSERT INTO imports(created_at,report) VALUES (?,?)",
                (datetime.now(UTC).isoformat(), encode(report)),
            )
        return report

    def snapshot(self):
        with self.connect() as db:
            assets = [
                json.loads(row[0])["asset"]
                for row in db.execute("SELECT normalized FROM assets ORDER BY id")
            ]
            events = [
                json.loads(row[0])
                for row in db.execute("SELECT payload FROM events ORDER BY id")
            ]
            decisions = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM decisions ORDER BY event_id,version"
                )
            ]
            count = db.execute("SELECT count(*) FROM observations").fetchone()[0]
            report = db.execute(
                "SELECT report FROM imports ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return {
            "schema": SCHEMA,
            "assets": assets,
            "events": events,
            "decisions": decisions,
            "observation_count": count,
            "last_import": json.loads(report[0]) if report else None,
        }

    def save_decision(self, value):
        request = DecisionInput.model_validate(value)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event, sha = self._evidence(db, request.event_id)
            current = db.execute(
                "SELECT coalesce(max(version),0) FROM decisions WHERE event_id=?",
                (request.event_id,),
            ).fetchone()[0]
            if request.expected_version != current:
                raise ValueError("Revision conflict: reload before saving")
            if not set(request.evidence_consulted) <= set(event.evidence):
                raise ValueError("Consulted evidence is not bound to this event")
            decision = ReviewDecision(
                **request.model_dump(),
                decision_id=str(uuid.uuid4()),
                version=current + 1,
                created_at=datetime.now(UTC),
                evidence_sha256=sha,
            )
            payload = decision.model_dump(mode="json")
            db.execute(
                "INSERT INTO decisions VALUES (?,?,?,?)",
                (
                    decision.decision_id,
                    decision.event_id,
                    decision.version,
                    encode(payload),
                ),
            )
        return payload

    @staticmethod
    def _evidence(db, event_id):
        row = db.execute(
            "SELECT e.payload,a.sha FROM events e JOIN assets a ON a.id=e.asset_id WHERE e.id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown event reference")
        return CandidateEvent.model_validate_json(row[0]), row[1]

    def import_decisions(self, path: Path):
        if path.suffix.lower() == ".csv":
            items = [
                json.loads(row["decision_json"])
                for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
            ]
        else:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc["schema"] != SCHEMA:
                raise ValueError("Unsupported decision schema")
            items = doc["decisions"]
        decisions = [ReviewDecision.model_validate(item) for item in items]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for d in sorted(decisions, key=lambda d: (d.event_id, d.version)):
                event, sha = self._evidence(db, d.event_id)
                if sha != d.evidence_sha256 or not set(d.evidence_consulted) <= set(
                    event.evidence
                ):
                    raise ValueError("Evidence binding mismatch")
                payload = encode(d.model_dump(mode="json"))
                old = db.execute(
                    "SELECT payload FROM decisions WHERE id=?", (d.decision_id,)
                ).fetchone()
                if old:
                    if old[0] != payload:
                        raise ValueError("Conflicting immutable decision")
                    continue
                current = db.execute(
                    "SELECT coalesce(max(version),0) FROM decisions WHERE event_id=?",
                    (d.event_id,),
                ).fetchone()[0]
                if d.version != current + 1:
                    raise ValueError("Missing or conflicting review history")
                db.execute(
                    "INSERT INTO decisions VALUES (?,?,?,?)",
                    (d.decision_id, d.event_id, d.version, payload),
                )
        return len(decisions)

    def export(self, format="json"):
        data = self.snapshot()
        if format == "json":
            return encode({"schema": SCHEMA, "decisions": data["decisions"]}).encode(
                "utf-8"
            )
        if format != "csv":
            raise ValueError("Export format must be json or csv")
        stream = io.StringIO(newline="")
        fields = [
            "event_id",
            "version",
            "status",
            "continuity",
            "dependence",
            "confidence",
            "reviewer",
            "created_at",
            "decision_json",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in data["decisions"]:
            row = {key: item[key] for key in fields if key != "decision_json"}
            # Spreadsheet formula injection protection; exact values live in JSON column.
            row = {
                k: "'" + v if isinstance(v, str) and v[:1] in "=+-@" else v
                for k, v in row.items()
            }
            row["decision_json"] = encode(item)
            writer.writerow(row)
        return stream.getvalue().encode("utf-8")

    def media(self, event_id: str, index: int):
        with self.connect() as db:
            event, sha = self._evidence(db, event_id)
            if index < 0 or index >= len(event.evidence):
                raise ValueError("Unknown media reference")
            raw = db.execute(
                "SELECT raw FROM assets WHERE id=?", (event.asset_id,)
            ).fetchone()[0]
        if digest(raw) != sha:
            raise ValueError("Frozen evidence checksum mismatch")
        with ZipFile(io.BytesIO(raw)) as archive:
            name = event.evidence[index]
            return archive.read(name), "image/png" if name.lower().endswith(
                ".png"
            ) else "image/jpeg"

    def verify(self, originals=False):
        errors = []
        with self.connect() as db:
            if (
                db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or db.execute("PRAGMA foreign_key_check").fetchall()
            ):
                errors.append("SQLite integrity failure")
            for aid, sha, origin, raw, stored in db.execute("SELECT * FROM assets"):
                try:
                    frozen = json.loads(stored)
                    if digest(raw) != sha:
                        raise ValueError("Frozen SHA256 mismatch")
                    asset, events, observations, media, _ = normalize(
                        raw, frozen["asset"]["filename"]
                    )
                    if (
                        asset != frozen["asset"]
                        or media != frozen["media"]
                        or aid != asset["asset_id"]
                    ):
                        raise ValueError("Normalized asset mismatch")
                    # CSV companions intentionally have no independently normalized rows.
                    if not frozen.get(
                        "csv_companion",
                        asset["kind"] == "csv" and not frozen["observations"],
                    ):
                        if (
                            events != frozen["events"]
                            or observations != frozen["observations"]
                        ):
                            raise ValueError("Normalization replay mismatch")
                    else:
                        if (
                            asset["kind"] != "csv"
                            or frozen["events"]
                            or frozen["observations"]
                        ):
                            raise ValueError("Invalid CSV companion role")
                        companion_name = str(
                            Path(asset["filename"]).with_suffix(".json")
                        )
                        matched = False
                        for primary_raw, primary_normalized in db.execute(
                            "SELECT raw,normalized FROM assets"
                        ):
                            primary = json.loads(primary_normalized)["asset"]
                            if primary["filename"] != companion_name:
                                continue
                            try:
                                compare_records(
                                    decode(primary_raw, ".json"), decode(raw, ".csv")
                                )
                                matched = True
                            except ValueError:
                                continue
                        if not matched:
                            raise ValueError(
                                "Missing or inconsistent JSON companion reference"
                            )
                    for table, key in [
                        ("events", "events"),
                        ("observations", "observations"),
                    ]:
                        actual = []
                        for rid, payload in db.execute(
                            f"SELECT id,payload FROM {table} WHERE asset_id=? ORDER BY id",
                            (aid,),
                        ):
                            record = json.loads(payload)
                            if (
                                rid
                                != record[
                                    "event_id" if key == "events" else "observation_id"
                                ]
                            ):
                                raise ValueError("Record primary key mismatch")
                            actual.append(record)
                        expected = sorted(
                            frozen[key],
                            key=lambda r: r[
                                "event_id" if key == "events" else "observation_id"
                            ],
                        )
                        if actual != expected:
                            raise ValueError(
                                "Stored records disagree with frozen evidence"
                            )
                    if originals and digest(Path(origin).read_bytes()) != sha:
                        raise ValueError("Original SHA256 mismatch")
                except (ValueError, OSError) as exc:
                    errors.append(f"{aid}: {exc}")
            versions = {}
            for did, eid, version, payload in db.execute(
                "SELECT id,event_id,version,payload FROM decisions ORDER BY event_id,version"
            ):
                try:
                    d = ReviewDecision.model_validate_json(payload)
                    if (did, eid, version) != (d.decision_id, d.event_id, d.version):
                        raise ValueError("Decision primary key mismatch")
                    event, sha = self._evidence(db, d.event_id)
                    if (
                        d.evidence_sha256 != sha
                        or d.version != versions.get(d.event_id, 0) + 1
                        or not set(d.evidence_consulted) <= set(event.evidence)
                    ):
                        raise ValueError("Decision binding/history invalid")
                    versions[d.event_id] = d.version
                except ValueError as exc:
                    errors.append(str(exc))
        return {
            "schema": SCHEMA,
            "valid": not errors,
            "errors": errors,
            "originals_checked": originals,
        }


def atomic_export(path: Path, data: bytes):
    if path.exists():
        raise ValueError("Export target already exists; choose a new filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".audit-export-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
