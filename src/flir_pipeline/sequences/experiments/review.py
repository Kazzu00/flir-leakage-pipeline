"""Read-only contact sheets and immutable external review history."""

import html
import io
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PIL import __version__ as pillow_version
from pydantic import Field

from flir_pipeline.features.image_source import ImageSource
from flir_pipeline.sequences.experiments.artifacts import (
    binding,
    inspect,
    publish,
    tables,
)
from flir_pipeline.sequences.experiments.config import StrictModel
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id


class Decision(StrictModel):
    review_query_id: str = Field(min_length=1)
    decision: Literal["supported", "ambiguous", "unsupported"]
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    notes: str


def context_plan(evidence, source, radius):
    meta = inspect(evidence)
    data = tables(evidence)
    queries, plan = [], []

    def add(query, records, role):
        for r in records.sort_values(
            ["timeline_id", "position", "frame_id"]
        ).itertuples():
            plan.append(
                {
                    "review_query_id": query,
                    "frame_id": r.frame_id,
                    "content_id": r.content_id,
                    "timeline_id": r.timeline_id,
                    "position": int(r.position) if pd.notna(r.position) else None,
                    "role": role,
                }
            )

    if meta["artifact_kind"] == "sequence_boundary_candidates_v2":
        for zone in data["candidate_zones"].itertuples():
            queries.append(
                {
                    "review_query_id": zone.candidate_id,
                    "kind": "boundary_zone_candidate",
                    "subject": f"{zone.timeline_id}: {zone.start}-{zone.end}",
                    "encoder_disagreement": zone.encoder_policy != "consensus",
                }
            )
            context = source.records.loc[
                source.records.timeline_id.eq(zone.timeline_id)
                & source.records.position.between(
                    zone.start - radius, zone.end + radius
                )
            ]
            add(zone.candidate_id, context, "wide_temporal_context")
    elif meta["artifact_kind"] == "sequence_recurrence_v1":
        pairs = data["pairs"]
        selected = pairs.loc[
            pairs.broad_candidate
            | pairs.encoder_disagreement
            | pairs.exact_copy_dependency_observed
        ]
        for pair in selected.itertuples():
            queries.append(
                {
                    "review_query_id": pair.pair_id,
                    "kind": "recurrence_pair_candidate",
                    "subject": f"{pair.left} / {pair.right}",
                    "encoder_disagreement": pair.encoder_disagreement,
                }
            )
            for encoder in ("clip", "dinov2"):
                matches = (
                    data["nearest_matches"]
                    .loc[
                        data["nearest_matches"].pair_id.eq(pair.pair_id)
                        & data["nearest_matches"].encoder.eq(encoder)
                    ]
                    .sort_values(
                        ["cosine", "query_content_id", "neighbor_content_id"],
                        ascending=[False, True, True],
                    )
                    .head(3)
                )
                ids = set(matches.query_content_id) | set(matches.neighbor_content_id)
                add(
                    pair.pair_id,
                    source.records.loc[source.records.content_id.isin(ids)],
                    f"strongest_{encoder}_matches",
                )
                for core in (pair.left, pair.right):
                    ids = sorted(
                        data["core_content_membership"].loc[
                            data["core_content_membership"].core_id.eq(core),
                            "content_id",
                        ]
                    )
                    indices = [source.contents.index(c) for c in ids]
                    x = source.embeddings[encoder][indices].astype(np.float64)
                    medoid = ids[int(np.argmax(x @ x.sum(axis=0)))]
                    add(
                        pair.pair_id,
                        source.records.loc[source.records.content_id.eq(medoid)],
                        f"{encoder}_medoid:{core}",
                    )
                    records = source.records.loc[
                        source.records.content_id.isin(ids)
                    ].sort_values(["position", "frame_id"])
                    # Explicit overview endpoints and evenly-spaced context, not
                    # a claim that a representative occurrence covers every copy.
                    chosen = records.iloc[
                        np.unique(
                            np.linspace(
                                0, len(records) - 1, min(9, len(records)), dtype=int
                            )
                        )
                    ]
                    add(pair.pair_id, chosen, f"core_overview:{core}")
            anchors = [
                p
                for p in plan
                if p["review_query_id"] == pair.pair_id
                and p["role"].startswith("strongest")
            ]
            for anchor in anchors:
                if anchor["position"] is not None:
                    context = source.records.loc[
                        source.records.timeline_id.eq(anchor["timeline_id"])
                        & source.records.position.between(
                            anchor["position"] - radius, anchor["position"] + radius
                        )
                    ]
                    context = context.sort_values(["position", "frame_id"])
                    add(
                        pair.pair_id,
                        context.iloc[
                            np.unique(
                                np.linspace(
                                    0, len(context) - 1, min(7, len(context)), dtype=int
                                )
                            )
                        ],
                        "wide_temporal_context",
                    )
    else:
        raise ValueError("Review package requires boundary or recurrence evidence")
    return (
        pd.DataFrame(
            queries,
            columns=["review_query_id", "kind", "subject", "encoder_disagreement"],
        ),
        pd.DataFrame(
            plan,
            columns=[
                "review_query_id",
                "frame_id",
                "content_id",
                "timeline_id",
                "position",
                "role",
            ],
        )
        .drop_duplicates()
        .sort_values(["review_query_id", "role", "timeline_id", "position", "frame_id"])
        .reset_index(drop=True),
    )


def create_package(
    evidence, source, output, images_root=None, images_archive=None, radius=20
):
    from flir_pipeline.linkage.review_storage import _safe_output

    evidence_paths = (
        list(evidence) if isinstance(evidence, (list, tuple)) else [evidence]
    )
    if not evidence_paths:
        raise ValueError("Review package needs evidence")
    protected = [*evidence_paths, *(source.paths or ())]
    protected.extend(p for p in (images_root, images_archive) if p is not None)
    _safe_output(output, protected)
    parts, support = [], {}
    for i, path in enumerate(evidence_paths):
        meta = inspect(path)
        if meta["identity"]["sources"].get("input") != source.signature:
            raise ValueError("Review evidence is not bound to the supplied source")
        parts.append(context_plan(path, source, radius))
        for name, table in tables(path).items():
            if name in {
                "pairs",
                "cluster_recurrence",
                "agreement",
                "disagreement",
                "candidate_zones",
                "nearest_matches",
            }:
                support[f"evidence_{i}_{name}"] = table
    queries = pd.concat([q for q, _ in parts], ignore_index=True).drop_duplicates()
    if not queries.review_query_id.is_unique:
        raise ValueError("Conflicting query IDs across combined evidence")
    plan = (
        pd.concat([p for _, p in parts], ignore_index=True)
        .drop_duplicates()
        .sort_values(["review_query_id", "role", "timeline_id", "position", "frame_id"])
        .reset_index(drop=True)
    )
    sources = {"input": source.signature}
    if len(evidence_paths) == 1:
        sources["evidence"] = binding(evidence_paths[0])
    else:
        sources["evidence_set"] = [binding(p) for p in evidence_paths]
    media, links = {}, []
    records = source.records.set_index("frame_id")
    with ImageSource(images_archive, images_root) as images:
        raw = source.records.copy()
        if "source_member_path" not in raw and "image_member_path" in raw:
            raw["source_member_path"] = raw.image_member_path
        column = images.path_column(raw)
        records = raw.set_index("frame_id")
        for query, group in plan.groupby("review_query_id", sort=True):
            for page, start in enumerate(range(0, len(group), 12)):
                sheet = Image.new("RGB", (960, 800), "#f4f4f4")
                draw = ImageDraw.Draw(sheet)
                draw.text(
                    (12, 8),
                    f"{query} | manual evidence, not ground truth",
                    fill="black",
                )
                for j, row in enumerate(group.iloc[start : start + 12].itertuples()):
                    record = records.loc[row.frame_id]
                    picture = images.decode(record[column], record.image_sha256)
                    picture.thumbnail((230, 185))
                    x, y = (j % 4) * 240, 32 + (j // 4) * 250
                    sheet.paste(picture, (x + (230 - picture.width) // 2, y))

                    # Full identifiers remain in context.parquet; captions must
                    # stay inside their tile even for long source-video hashes.
                    def caption_id(value):
                        text = str(value)
                        return text if len(text) <= 34 else text[:31] + "..."

                    caption = "\n".join(
                        caption_id(s)
                        for s in (
                            f"{row.timeline_id} / {row.position}",
                            row.role,
                            row.frame_id,
                        )
                    )
                    draw.multiline_text((x + 5, y + 188), caption, fill="black")
                buffer = io.BytesIO()
                sheet.save(buffer, format="PNG")
                name = f"media/{query}_{page:03d}.png"
                media[name] = buffer.getvalue()
                query_info = queries.set_index("review_query_id").loc[query]
                links.append(
                    f"<h2>{html.escape(query_info.subject)}</h2><p>{html.escape(query)} | "
                    f"encoder disagreement: {bool(query_info.encoder_disagreement)}</p>"
                    f'<img src="{html.escape(Path(name).name)}" alt="Review context">'
                )
    decisions = pd.DataFrame(
        {
            "review_query_id": queries.review_query_id,
            "decision": "",
            "reviewer": "",
            "reviewed_at": "",
            "notes": "",
        }
    )
    media["media/decisions_template.csv"] = decisions.to_csv(
        index=False, lineterminator="\n"
    ).encode()
    media["media/index.html"] = (
        '<!doctype html><meta charset="utf-8"><title>Sequence evidence review</title>'
        "<h1>Evidencia para revisión manual</h1><p>Candidatos; sin confirmaciones automáticas, ground truth ni split.</p>"
        + "\n".join(links)
    ).encode()
    support_links = []
    for name, table in support.items():
        media[f"media/{name}.csv"] = table.to_csv(
            index=False, lineterminator="\n"
        ).encode()
        support_links.append(
            f'<li><a href="{name}.csv">{name}</a> ({len(table)} rows)</li>'
        )
    media["media/index.html"] += (
        "<h2>Evidence and cluster support</h2><ul>" + "".join(support_links) + "</ul>"
    ).encode()
    source.unchanged()
    return publish(
        output,
        "sequence_review_package_v1",
        {
            "context_radius": radius,
            "pillow": pillow_version,
            "sheet_policy": "12_images_per_page_all_selected_occurrences_v1",
        },
        sources,
        {
            "queries": queries,
            "context": plan,
            "occurrences": source.records.copy(),
            **support,
        },
        {
            "query_count": len(queries),
            "selected_context_occurrences": int(plan.frame_id.nunique()),
            "all_source_occurrences_preserved": True,
            "manual_review_required": not queries.empty,
        },
        media=media,
    )


def import_decisions(package, decisions, output, previous=None):
    from datetime import datetime

    package_meta = inspect(package)
    if package_meta["artifact_kind"] != "sequence_review_package_v1":
        raise ValueError("Expected a frozen review package")
    incoming = pd.read_csv(decisions, dtype=str, keep_default_na=False)
    if (
        list(incoming) != list(Decision.model_fields)
        or not incoming.review_query_id.is_unique
        or incoming.empty
    ):
        raise ValueError(
            "Invalid exact decision schema, duplicate decisions or empty import"
        )
    queries = tables(package)["queries"]
    if not set(incoming.review_query_id) <= set(queries.review_query_id):
        raise ValueError("Decision does not belong to this package")
    for row in incoming.to_dict("records"):
        value = Decision.model_validate(row)
        if (
            datetime.fromisoformat(value.reviewed_at.replace("Z", "+00:00")).tzinfo
            is None
        ):
            raise ValueError("Manual review timestamp requires a timezone")
    history, prior, media = [], incoming.iloc[:0].copy(), {}
    sources = {**package_meta["identity"]["sources"], "package": binding(package)}
    if previous:
        meta = inspect(previous)
        if meta["artifact_kind"] != "sequence_manual_review_v1" or meta["identity"][
            "sources"
        ]["package"] != binding(package):
            raise ValueError("Previous revision belongs to another package")
        prior = tables(previous)["decisions"]
        history = read_json(previous / "decision_history.json")["events"]
        verify_decision_history(previous)
        media = {
            name: (previous / name).read_bytes() for name in meta["identity"]["media"]
        }
        sources["previous"] = binding(previous)
    for row in incoming.itertuples():
        existing = prior.loc[prior.review_query_id.eq(row.review_query_id)]
        if len(existing) and existing.decision.iloc[0] != row.decision:
            raise ValueError(
                "Conflicting decision; preserve history and adjudicate externally"
            )
    event = {
        "import_sha256": file_sha256(decisions),
        "decisions": incoming.sort_values("review_query_id").to_dict("records"),
    }
    event["event_id"] = stable_id(event)
    if any(existing["event_id"] == event["event_id"] for existing in history):
        return previous
    media[f"media/imports/{event['event_id']}.csv"] = decisions.read_bytes()
    history = [*history, event]
    merged = (
        pd.concat([prior, incoming])
        .drop_duplicates("review_query_id", keep="first")
        .sort_values("review_query_id")
        .reset_index(drop=True)
    )
    result = publish(
        output,
        "sequence_manual_review_v1",
        {"decision_policy": "external_no_automatic_confirmation"},
        sources,
        {"decisions": merged, "queries": queries},
        {
            "reviewed_queries": len(merged),
            "total_queries": len(queries),
            "history_events": len(history),
        },
        extras={"decision_history": {"events": history}},
        media=media,
    )
    verify_decision_history(result)
    return result


def verify_decision_history(directory):
    """Replay only explicit decisions; no status can be inferred from imagery."""
    meta = inspect(directory)
    if meta["artifact_kind"] != "sequence_manual_review_v1":
        raise ValueError("Expected manual evidence revision")
    history = read_json(directory / "decision_history.json")["events"]
    seen, rows = set(), {}
    valid_queries = set(pd.read_parquet(directory / "queries.parquet").review_query_id)
    for event in history:
        if (
            set(event) != {"event_id", "import_sha256", "decisions"}
            or event["event_id"]
            != stable_id({k: v for k, v in event.items() if k != "event_id"})
            or event["event_id"] in seen
        ):
            raise ValueError("Invalid review history event identity")
        seen.add(event["event_id"])
        path = directory / f"media/imports/{event['event_id']}.csv"
        if file_sha256(path) != event["import_sha256"]:
            raise ValueError("Changed frozen manual import")
        snapshot = pd.read_csv(path, dtype=str, keep_default_na=False)
        if (
            list(snapshot) != list(Decision.model_fields)
            or not snapshot.review_query_id.is_unique
            or snapshot.sort_values("review_query_id").to_dict("records")
            != event["decisions"]
        ):
            raise ValueError("History differs from frozen CSV")
        for row in event["decisions"]:
            Decision.model_validate(row)
            query = row["review_query_id"]
            if query not in valid_queries or (
                query in rows and rows[query]["decision"] != row["decision"]
            ):
                raise ValueError("Unknown/conflicting decision in history")
            rows.setdefault(query, row)
    expected = pd.DataFrame(
        [rows[k] for k in sorted(rows)], columns=list(Decision.model_fields)
    )
    pd.testing.assert_frame_equal(
        expected, pd.read_parquet(directory / "decisions.parquet")
    )
    return {
        "quality_valid": True,
        "replayed_events": len(history),
        "ground_truth": False,
        "split_created": False,
        "automatic_confirmation": False,
    }
