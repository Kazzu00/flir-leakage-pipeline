"""Transactional local ledger. Raw inputs and append-only decisions share one DB.

Changed source revisions require a separate ledger and never silently inherit
a judgment. SQLite commits make interruption rollback atomic; optimistic
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
    historical_anomalies,
    normalize,
)
from .legacy_importers import normalize as normalize_v1
from .models import (
    REVISION,
    SCHEMA,
    CandidateEvent,
    DecisionInput,
    ExternalEvidence,
    ReviewDecision,
    require_evidence,
)


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
                CREATE TABLE IF NOT EXISTS companions (
                    asset_id TEXT PRIMARY KEY REFERENCES assets(id),
                    canonical_asset_id TEXT NOT NULL REFERENCES assets(id), receipt TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS external_evidence (
                    id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(id),
                    raw BLOB NOT NULL, payload TEXT NOT NULL);
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
            (
                p
                for p in root.iterdir()
                if p.suffix.lower() in {".json", ".csv", ".zip"}
            ),
            key=lambda p: (p.suffix.lower() != ".json", p.name),
        )
        report["excluded_by_priority_selection"] = [
            p.name for p in paths if priority_only and p.name not in PRIORITY
        ]
        if priority_only:
            paths = [p for p in paths if p.name in PRIORITY]
        for path in paths:
            entry = {"file": path.name}
            try:
                raw = bounded_bytes(path)
                entry["sha256"] = digest(raw)
                asset, events, observations, media, warnings = normalize(raw, path.name)
                if path.suffix in {".json", ".csv"}:
                    other_path = path.with_suffix(
                        ".csv" if path.suffix == ".json" else ".json"
                    )
                    if other_path.is_file():
                        other_raw = bounded_bytes(other_path)
                        normalize(other_raw, other_path.name)
                        compare_pair(path.name, raw, other_raw)
                frozen = {
                    "asset": asset,
                    "events": events,
                    "observations": observations,
                    "media": media,
                    "normalizer": "1.1",
                }
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    existing = db.execute(
                        "SELECT id FROM assets WHERE id=?", (asset["asset_id"],)
                    ).fetchone()
                    canonical_id = None
                    if not existing:
                        for aid, origin, previous_raw, previous in db.execute(
                            "SELECT id,origin,raw,normalized FROM assets ORDER BY rowid"
                        ):
                            previous = json.loads(previous)["asset"]
                            if Path(origin).parent != root:
                                continue
                            if previous["filename"] == path.name:
                                raise ValueError(
                                    "Source revision conflict: changed bytes for an existing filename; use a separate ledger"
                                )
                            if (
                                path.suffix in {".json", ".csv"}
                                and previous["kind"] in {"json", "csv"}
                                and Path(previous["filename"]).stem == path.stem
                            ):
                                compare_pair(path.name, raw, previous_raw)
                                canonical_id = self._companion_map(db).get(aid, aid)
                        db.execute(
                            "INSERT INTO assets VALUES (?,?,?,?,?)",
                            (
                                asset["asset_id"],
                                asset["sha256"],
                                str(path.resolve()),
                                raw,
                                encode(frozen),
                            ),
                        )
                        if canonical_id:
                            receipt = {
                                "operation": "equivalent_companion",
                                "policy": "first_accepted_source_keeps_ids",
                                "asset_id": asset["asset_id"],
                                "canonical_asset_id": canonical_id,
                                "created_at": datetime.now(UTC).isoformat(),
                                "decisions_migrated": False,
                            }
                            db.execute(
                                "INSERT INTO companions VALUES (?,?,?)",
                                (asset["asset_id"], canonical_id, encode(receipt)),
                            )
                        else:
                            for table, items, key in [
                                ("events", events, "event_id"),
                                ("observations", observations, "observation_id"),
                            ]:
                                db.executemany(
                                    f"INSERT INTO {table} VALUES (?,?,?)",
                                    [
                                        (r[key], asset["asset_id"], encode(r))
                                        for r in items
                                    ],
                                )
                    else:
                        canonical_id = self._companion_map(db).get(asset["asset_id"])
                if canonical_id:
                    warnings.append(
                        "Equivalent companion retained; canonical IDs and reviews unchanged."
                    )
                entry.update(
                    status="valid",
                    events=0 if canonical_id else len(events),
                    observations=0 if canonical_id else len(observations),
                    warnings=warnings,
                    canonical_asset_id=canonical_id or asset["asset_id"],
                    anomalies=historical_anomalies(observations),
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

    @staticmethod
    def _companion_map(db):
        mapping = dict(db.execute("SELECT asset_id,canonical_asset_id FROM companions"))
        # v1 CSV companions have no event rows. Recognize them read-only; do not
        # alter the original normalized snapshot or any review payload.
        assets = list(
            db.execute("SELECT id,origin,raw,normalized FROM assets ORDER BY rowid")
        )
        for aid, origin, raw, stored in assets:
            frozen = json.loads(stored)
            if aid in mapping or not frozen.get(
                "csv_companion",
                frozen["asset"]["kind"] == "csv"
                and not frozen["observations"]
                and not frozen["events"],
            ):
                continue
            for other_id, other_origin, other_raw, other_stored in assets:
                other = json.loads(other_stored)["asset"]
                if Path(origin).parent == Path(other_origin).parent and other[
                    "filename"
                ] == str(Path(frozen["asset"]["filename"]).with_suffix(".json")):
                    compare_pair(frozen["asset"]["filename"], raw, other_raw)
                    mapping[aid] = other_id
                    break
        return mapping

    def snapshot(self):
        with self.connect() as db:
            mapping = self._companion_map(db)
            assets, events, anomalies = [], [], []
            projections = {}
            for aid, raw, stored in db.execute(
                "SELECT id,raw,normalized FROM assets ORDER BY rowid"
            ):
                frozen = json.loads(stored)
                assets.append(frozen["asset"])
                _, projected, observations, _, _ = normalize(
                    raw, frozen["asset"]["filename"]
                )
                projections[aid] = projected
                if aid not in mapping:
                    events.extend(projected)
                    anomalies.extend(historical_anomalies(observations))
            # Supplement temporal metadata from equivalent JSON without changing
            # canonical identity, asset binding or previous decisions.
            companions = []
            for aid, primary in mapping.items():
                companions.append(
                    {
                        "asset_id": aid,
                        "canonical_asset_id": primary,
                        "decisions_migrated": False,
                        "event_links": [
                            {
                                "canonical_event_id": a["event_id"],
                                "companion_event_id": b["event_id"],
                            }
                            for a, b in zip(
                                projections[primary], projections[aid], strict=True
                            )
                        ],
                        "receipt": next(
                            (
                                json.loads(r[0])
                                for r in db.execute(
                                    "SELECT receipt FROM companions WHERE asset_id=?",
                                    (aid,),
                                )
                            ),
                            None,
                        ),
                    }
                )
                for canonical, supplement in zip(
                    projections[primary], projections[aid], strict=True
                ):
                    if (
                        supplement["sample_rate_fps"] is not None
                        and canonical["sample_rate_fps"] is None
                    ):
                        for field in (
                            "event_start_sample",
                            "event_end_sample",
                            "start_second",
                            "end_second",
                            "sample_rate_fps",
                            "temporal_basis",
                        ):
                            canonical[field] = supplement[field]
                        canonical["temporal_basis"] += "; supplemental_asset=" + aid
            decisions = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM decisions ORDER BY event_id,version"
                )
            ]
            count = sum(
                1
                for (aid,) in db.execute("SELECT asset_id FROM observations")
                if aid not in mapping
            )
            report = db.execute(
                "SELECT report FROM imports ORDER BY id DESC LIMIT 1"
            ).fetchone()
            external = [
                json.loads(r[0])
                for r in db.execute("SELECT payload FROM external_evidence ORDER BY id")
            ]
        legacy_warnings = []
        for value in decisions:
            try:
                require_evidence(ReviewDecision.model_validate(value))
            except ValueError:
                legacy_warnings.append(value["decision_id"])
        return {
            "schema": SCHEMA,
            "contract_revision": REVISION,
            "assets": assets,
            "events": events,
            "decisions": decisions,
            "observation_count": count,
            "last_import": json.loads(report[0]) if report else None,
            "external_evidence": external,
            "companions": companions,
            "anomalies": anomalies,
            "legacy_review_warnings": legacy_warnings,
        }

    def save_decision(self, value):
        request = DecisionInput.model_validate(value)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event, sha = self._evidence(db, request.event_id)
            if event.asset_id in self._companion_map(db):
                raise ValueError(
                    "This event is a companion; review its canonical event explicitly"
                )
            self._validate_external(db, request)
            current = db.execute(
                "SELECT coalesce(max(version),0) FROM decisions WHERE event_id=?",
                (request.event_id,),
            ).fetchone()[0]
            if request.expected_version != current:
                raise ValueError("Revision conflict: reload before saving")
            if not set(request.evidence_consulted) <= set(event.evidence):
                raise ValueError("Consulted evidence is not bound to this event")
            self._validate_native(db, request, event, sha)
            decision = ReviewDecision(
                **request.model_dump(),
                policy_version="1.1",
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
        decisions = [(ReviewDecision.model_validate(item), item) for item in items]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for d, original in sorted(
                decisions, key=lambda pair: (pair[0].event_id, pair[0].version)
            ):
                event, sha = self._evidence(db, d.event_id)
                if sha != d.evidence_sha256 or not set(d.evidence_consulted) <= set(
                    event.evidence
                ):
                    raise ValueError("Evidence binding mismatch")
                payload = encode(original)
                old = db.execute(
                    "SELECT payload FROM decisions WHERE id=?", (d.decision_id,)
                ).fetchone()
                if old:
                    if old[0] != payload:
                        raise ValueError("Conflicting immutable decision")
                    continue
                require_evidence(d)
                self._validate_external(db, d)
                if event.asset_id in self._companion_map(db):
                    raise ValueError(
                        "New reviews cannot target a reconciled companion identity"
                    )
                self._validate_native(db, d, event, sha)
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
            return encode(
                {
                    "schema": SCHEMA,
                    "contract_revision": REVISION,
                    "decisions": data["decisions"],
                }
            ).encode("utf-8")
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

    @staticmethod
    def _validate_native(db, decision, event, sha):
        if not decision.evidence_consulted:
            return
        raw = db.execute(
            "SELECT raw FROM assets WHERE id=?", (event.asset_id,)
        ).fetchone()[0]
        if digest(raw) != sha:
            raise ValueError("Consulted evidence checksum mismatch")
        try:
            with ZipFile(io.BytesIO(raw)) as archive:
                for name in decision.evidence_consulted:
                    if archive.getinfo(name).file_size > 16 * 1024 * 1024:
                        raise ValueError("Consulted image exceeds limit")
                    verify_image(archive.read(name))
        except (OSError, BadZipFile, KeyError) as exc:
            raise ValueError(
                "Consulted visual evidence is unavailable or unreadable"
            ) from exc

    @staticmethod
    def _validate_external(db, decision):
        for evidence_id in decision.external_evidence_consulted:
            row = db.execute(
                "SELECT raw,payload FROM external_evidence WHERE id=? AND event_id=?",
                (evidence_id, decision.event_id),
            ).fetchone()
            if row is None:
                raise ValueError("External evidence is not registered for this event")
            evidence = ExternalEvidence.model_validate_json(row[1])
            _, sha = Store._evidence(db, decision.event_id)
            if (
                digest(row[0]) != evidence.sha256
                or evidence.event_asset_sha256 != sha
                or evidence_identity(evidence) != evidence_id
            ):
                raise ValueError("External evidence checksum/binding mismatch")

    def attach_evidence(self, event_id, path, *, locator, basis, rationale, reviewer):
        from PIL import Image

        if not all(v.strip() for v in (locator, rationale, reviewer)):
            raise ValueError(
                "Evidence association requires locator, rationale and reviewer"
            )
        raw = bounded_bytes(Path(path), limit=16 * 1024 * 1024)
        with Image.open(io.BytesIO(raw)) as picture:
            if (
                picture.format not in {"JPEG", "PNG"}
                or picture.width * picture.height > 40_000_000
            ):
                raise ValueError("Only bounded PNG/JPEG visual evidence is accepted")
            mime = "image/png" if picture.format == "PNG" else "image/jpeg"
            picture.verify()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event, sha = self._evidence(db, event_id)
            if event.asset_id in self._companion_map(db):
                raise ValueError("Attach evidence to the canonical event explicitly")
            binding = {
                "event_id": event_id,
                "sha256": digest(raw),
                "event_asset_sha256": sha,
                "filename": Path(path).name,
                "locator": locator,
                "basis": basis,
                "rationale": rationale,
                "reviewer": reviewer,
                "mime": mime,
            }
            eid = digest(encode(binding).encode())
            evidence = ExternalEvidence(
                evidence_id=eid, created_at=datetime.now(UTC), **binding
            )
            old = db.execute(
                "SELECT payload FROM external_evidence WHERE id=?", (eid,)
            ).fetchone()
            if old:
                return json.loads(old[0])
            payload = evidence.model_dump(mode="json")
            db.execute(
                "INSERT INTO external_evidence VALUES (?,?,?,?)",
                (eid, event_id, raw, encode(payload)),
            )
        return payload

    def external_media(self, evidence_id):
        with self.connect() as db:
            row = db.execute(
                "SELECT raw,payload FROM external_evidence WHERE id=?", (evidence_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Unknown external evidence")
        evidence = ExternalEvidence.model_validate_json(row[1])
        if digest(row[0]) != evidence.sha256:
            raise ValueError("External evidence checksum mismatch")
        return row[0], evidence.mime

    def reconcile_companions(
        self, canonical_asset_id, companion_asset_id, reviewer, rationale
    ):
        """Explicit v1 duplicate repair: keep physical records/history, add a receipt.

        Review branches are not merged. The UI discloses retained alternate
        histories and all future writes address the canonical identity explicitly.
        """
        if (
            not reviewer.strip()
            or not rationale.strip()
            or canonical_asset_id == companion_asset_id
        ):
            raise ValueError(
                "Explicit reviewer, rationale and distinct source identities required"
            )
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = {
                r[0]: r
                for r in db.execute(
                    "SELECT id,origin,raw,normalized FROM assets WHERE id IN (?,?)",
                    (canonical_asset_id, companion_asset_id),
                )
            }
            if len(rows) != 2:
                raise ValueError("Unknown reconciliation source")
            first, second = rows[canonical_asset_id], rows[companion_asset_id]
            a, b = json.loads(first[3])["asset"], json.loads(second[3])["asset"]
            if (
                {a["kind"], b["kind"]} != {"json", "csv"}
                or Path(first[1]).parent != Path(second[1]).parent
                or Path(a["filename"]).stem != Path(b["filename"]).stem
            ):
                raise ValueError("Not a scoped JSON/CSV companion pair")
            compare_pair(a["filename"], first[2], second[2])
            mapping = self._companion_map(db)
            if (
                canonical_asset_id in mapping
                or companion_asset_id in mapping
                or companion_asset_id in mapping.values()
            ):
                raise ValueError("Already linked or noncanonical reconciliation source")
            receipt = {
                "operation": "explicit_legacy_reconciliation",
                "canonical_asset_id": canonical_asset_id,
                "asset_id": companion_asset_id,
                "reviewer": reviewer,
                "rationale": rationale,
                "created_at": datetime.now(UTC).isoformat(),
                "decisions_migrated": False,
            }
            db.execute(
                "INSERT INTO companions VALUES (?,?,?)",
                (companion_asset_id, canonical_asset_id, encode(receipt)),
            )
        return receipt

    def verify(self, originals=False):
        errors, warnings = [], []
        with self.connect() as db:
            if (
                db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
                or db.execute("PRAGMA foreign_key_check").fetchall()
            ):
                errors.append("SQLite integrity failure")
            try:
                mapping = self._companion_map(db)
            except (ValueError, KeyError) as exc:
                mapping = {}
                errors.append(str(exc))
            source_rows = list(
                db.execute(
                    "SELECT id,sha,origin,raw,normalized FROM assets ORDER BY rowid"
                )
            )
            sources = {r[0]: r for r in source_rows}
            for aid, sha, origin, raw, stored in source_rows:
                try:
                    frozen = json.loads(stored)
                    if digest(raw) != sha:
                        raise ValueError("Frozen SHA256 mismatch")
                    normalizer = (
                        normalize if frozen.get("normalizer") == "1.1" else normalize_v1
                    )
                    asset, events, observations, media, _ = normalizer(
                        raw, frozen["asset"]["filename"]
                    )
                    if (
                        asset != frozen["asset"]
                        or aid != asset["asset_id"]
                        or media != frozen["media"]
                    ):
                        raise ValueError("Normalized asset mismatch")
                    legacy_empty = frozen.get(
                        "csv_companion",
                        frozen["asset"]["kind"] == "csv"
                        and not frozen["observations"]
                        and not frozen["events"],
                    )
                    if legacy_empty:
                        if (
                            aid not in mapping
                            or frozen["events"]
                            or frozen["observations"]
                        ):
                            raise ValueError("Missing or invalid legacy companion")
                        events, observations = [], []
                    if (
                        events != frozen["events"]
                        or observations != frozen["observations"]
                    ):
                        raise ValueError("Normalization replay mismatch")
                    # New alternate sources retain full normalization in the
                    # snapshot but have no duplicate canonical rows in tables.
                    is_new_alternate = (
                        aid in mapping and frozen.get("normalizer") == "1.1"
                    )
                    for table, values, key in [
                        ("events", events, "event_id"),
                        ("observations", observations, "observation_id"),
                    ]:
                        expected = (
                            []
                            if is_new_alternate
                            else sorted(values, key=lambda r: r[key])
                        )
                        actual = []
                        for rid, payload in db.execute(
                            f"SELECT id,payload FROM {table} WHERE asset_id=? ORDER BY id",
                            (aid,),
                        ):
                            record = json.loads(payload)
                            if rid != record[key]:
                                raise ValueError("Record primary key mismatch")
                            actual.append(record)
                        if actual != expected:
                            raise ValueError(
                                "Stored records disagree with frozen evidence"
                            )
                    if originals and digest(bounded_bytes(Path(origin))) != sha:
                        raise ValueError("Original SHA256 mismatch")
                    if aid in mapping:
                        target = mapping[aid]
                        if target not in sources or target in mapping:
                            raise ValueError("Invalid companion target/chain")
                        other = sources[target]
                        if Path(origin).parent != Path(other[2]).parent:
                            raise ValueError("Companion provenance scope mismatch")
                        other_asset = json.loads(other[4])["asset"]
                        if {asset["kind"], other_asset["kind"]} != {
                            "json",
                            "csv",
                        } or Path(asset["filename"]).stem != Path(
                            other_asset["filename"]
                        ).stem:
                            raise ValueError("Invalid companion types or filenames")
                        compare_pair(asset["filename"], raw, other[3])
                    for other in source_rows:
                        if (
                            other[0] == aid
                            or Path(origin).parent != Path(other[2]).parent
                        ):
                            continue
                        other_asset = json.loads(other[4])["asset"]
                        if {asset["kind"], other_asset["kind"]} == {
                            "json",
                            "csv",
                        } and Path(asset["filename"]).stem == Path(
                            other_asset["filename"]
                        ).stem:
                            compare_pair(asset["filename"], raw, other[3])
                            if mapping.get(aid, aid) != mapping.get(other[0], other[0]):
                                raise ValueError(
                                    "Duplicate canonical JSON/CSV records: explicit reconciliation required"
                                )
                except (
                    ValueError,
                    OSError,
                    KeyError,
                    TypeError,
                    BadZipFile,
                    csv.Error,
                ) as exc:
                    errors.append(f"{aid}: {exc}")
            versions = {}
            latest_ungrounded = {}
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
                    self._validate_external(db, d)
                    self._validate_native(db, d, event, sha)
                    versions[d.event_id] = d.version
                    try:
                        require_evidence(d)
                        latest_ungrounded[d.event_id] = False
                    except ValueError:
                        latest_ungrounded[d.event_id] = True
                        warnings.append(
                            f"Legacy ungrounded review preserved, requires explicit new review: {did}"
                        )
                except ValueError as exc:
                    errors.append(str(exc))
            for eid, event_id, raw, payload in db.execute(
                "SELECT * FROM external_evidence"
            ):
                try:
                    evidence = ExternalEvidence.model_validate_json(payload)
                    _, sha = self._evidence(db, event_id)
                    if (
                        evidence.evidence_id != eid
                        or evidence.event_id != event_id
                        or evidence.sha256 != digest(raw)
                        or evidence.event_asset_sha256 != sha
                        or evidence_identity(evidence) != eid
                    ):
                        raise ValueError("External evidence binding/hash invalid")
                except ValueError as exc:
                    errors.append(str(exc))
        return {
            "schema": SCHEMA,
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "originals_checked": originals,
            "latest_reviews_evidence_valid": not errors
            and not any(latest_ungrounded.values()),
        }


def bounded_bytes(path, limit=MAX_FILE):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Not a regular bounded artifact")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Artifact exceeds size limit")
    return raw


def verify_image(raw):
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as picture:
            if (
                picture.format not in {"JPEG", "PNG"}
                or picture.width * picture.height > 40_000_000
            ):
                raise ValueError("Unsupported or oversized visual evidence")
            picture.verify()
    except OSError as exc:
        raise ValueError("Unreadable visual evidence") from exc


def compare_pair(filename, raw, other_raw):
    if Path(filename).suffix == ".json":
        compare_records(decode(raw, ".json"), decode(other_raw, ".csv"))
    else:
        compare_records(decode(other_raw, ".json"), decode(raw, ".csv"))


def evidence_identity(evidence):
    binding = evidence.model_dump(mode="json", exclude={"evidence_id", "created_at"})
    return digest(encode(binding).encode())


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
