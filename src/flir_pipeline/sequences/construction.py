"""Reviewed inclusive intervals and sparse exact-copy dependency constraints."""

import numpy as np
import pandas as pd

from flir_pipeline.similarity.storage import stable_id

EDGE_COLUMNS = [
    "dependency_edge_id",
    "left_sequence_id",
    "right_sequence_id",
    "content_id",
    "left_occurrence_count",
    "right_occurrence_count",
]


def exact_dependencies(
    occurrences: pd.DataFrame, sequence_ids: list[str], sequence_set_id: str
):
    """A deterministic star per shared content has the same components as a clique.

    This keeps edges/support linear in occurrence membership even if one content
    repeats in every sequence. The support table retains ALL supporting frame IDs;
    no occurrence pairs or quadratic distances are materialized.
    """
    parent = {sid: sid for sid in sequence_ids}

    def find(sid):
        root = sid
        while parent[root] != root:
            root = parent[root]
        while parent[sid] != sid:
            previous = parent[sid]
            parent[sid] = root
            sid = previous
        return root

    edges, shared = [], []
    for content, group in occurrences.groupby("content_id", sort=True):
        counts = group.sequence_id.value_counts().sort_index()
        if len(counts) < 2:
            continue
        shared.append(content)
        left = counts.index[0]
        for right in counts.index[1:]:
            row = {
                "left_sequence_id": left,
                "right_sequence_id": right,
                "content_id": content,
                "left_occurrence_count": int(counts[left]),
                "right_occurrence_count": int(counts[right]),
            }
            edges.append(
                {
                    "dependency_edge_id": stable_id(
                        {
                            "kind": "exact_duplicate_dependency_edge",
                            "sequence_set_id": sequence_set_id,
                            **row,
                        }
                    ),
                    **row,
                }
            )
            a, b = sorted((find(left), find(right)))
            parent[b] = a
    components = {}
    for sid in sorted(sequence_ids):
        components.setdefault(find(sid), []).append(sid)
    membership = {}
    for members in components.values():
        gid = stable_id(
            {
                "kind": "exact_duplicate_dependency_group",
                "sequence_set_id": sequence_set_id,
                "sequence_ids": members,
            }
        )
        membership.update(dict.fromkeys(members, gid))
    edge_table = pd.DataFrame(edges, columns=EDGE_COLUMNS).astype(
        {name: "int64" if name.endswith("count") else "string" for name in EDGE_COLUMNS}
    )
    support_columns = [
        "content_id",
        "sequence_id",
        "frame_id",
        "video_id",
        "sample_index",
        "timestamp_seconds",
    ]
    support = occurrences.loc[occurrences.content_id.isin(shared), support_columns]
    support = support.sort_values(
        ["content_id", "sequence_id", "sample_index", "frame_id"]
    ).reset_index(drop=True)
    return edge_table, support, membership


def construct_tables(records, events, review, validation_signature, sequence_set_id):
    """Only confirmed accepts create cuts; exact repeated frames remain occurrences."""
    accepted = review.loc[review.decision.eq("accept")].sort_values(
        ["video_id", "localized_sample_index"]
    )
    boundaries = accepted.merge(
        events[["video_id", "coarse_sample_index", "candidate_id"]],
        on=["video_id", "coarse_sample_index"],
        validate="one_to_one",
    ).reset_index(drop=True)
    boundaries = boundaries.rename(
        columns={"sequence_boundary_committed": "input_sequence_boundary_committed"}
    )
    boundaries["boundary_id"] = pd.Series(
        [
            stable_id(
                {
                    "kind": "reviewed_sequence_boundary",
                    "sequence_set_id": sequence_set_id,
                    "video_id": row.video_id,
                    "sample_index": int(row.localized_sample_index),
                    "event_id": int(row.event_id),
                    "boundary_type": row.boundary_type,
                }
            )
            for row in boundaries.itertuples()
        ],
        dtype="string",
    )
    boundaries["validation_sha256"] = validation_signature["validation_sha256"]
    boundaries["ground_truth"] = False
    boundaries["sequence_boundary_committed"] = True
    boundaries["confirmation_timestamp_utc"] = validation_signature["metadata"][
        "confirmation_timestamp_utc"
    ]
    sequences, assigned = [], []
    for video, timeline in records.groupby("video_id", sort=True):
        timeline = timeline.sort_values("sample_index").reset_index(drop=True)
        cuts = boundaries.loc[boundaries.video_id.eq(video)].reset_index(drop=True)
        positions = cuts.localized_sample_index.to_numpy(dtype=np.int64)
        n = len(timeline)
        if len(positions) and (
            positions.min() <= 0
            or positions.max() >= n
            or np.any(np.diff(positions) <= 0)
        ):
            raise ValueError(
                "Accepted cuts must be unique, strictly increasing, interior and same-video"
            )
        endpoints = [0, *positions.tolist(), n]
        labels = np.empty(n, dtype=object)
        for i, (start, stop) in enumerate(
            zip(endpoints[:-1], endpoints[1:], strict=True)
        ):
            sid = stable_id(
                {
                    "kind": "sequence_instance",
                    "sequence_set_id": sequence_set_id,
                    "video_id": video,
                    "start_sample_index": start,
                    "end_sample_index": stop - 1,
                }
            )
            labels[start:stop] = sid
            first, last = timeline.iloc[start], timeline.iloc[stop - 1]
            sequences.append(
                {
                    "sequence_id": sid,
                    "sequence_set_id": sequence_set_id,
                    "video_id": video,
                    "source_video": first.source_video,
                    "source_video_sha256": first.source_video_sha256,
                    "start_sample_index": start,
                    "end_sample_index": stop - 1,
                    "occurrence_count": stop - start,
                    "start_timestamp_seconds": float(first.timestamp_seconds),
                    "end_timestamp_seconds": float(last.timestamp_seconds),
                    "sample_fps": 1.0,
                    "start_frame_id": first.frame_id,
                    "end_frame_id": last.frame_id,
                    "start_boundary_id": None
                    if i == 0
                    else cuts.iloc[i - 1].boundary_id,
                    "end_boundary_id": None
                    if i == len(cuts)
                    else cuts.iloc[i].boundary_id,
                    "start_boundary_type": "source_start"
                    if i == 0
                    else cuts.iloc[i - 1].boundary_type,
                    "validation_sha256": validation_signature["validation_sha256"],
                    "ground_truth": False,
                }
            )
        assigned.append(
            timeline.assign(sequence_id=labels, sequence_set_id=sequence_set_id)
        )
    occurrences = pd.concat(assigned, ignore_index=True)
    instances = pd.DataFrame(sequences)
    if not instances.sequence_id.is_unique:
        raise ValueError("Sequence identity collision")
    edges, support, membership = exact_dependencies(
        occurrences, instances.sequence_id.tolist(), sequence_set_id
    )
    for frame in (instances, occurrences):
        frame["exact_duplicate_dependency_group_id"] = frame.sequence_id.map(membership)
    return {
        "boundaries": boundaries,
        "sequence_instances": instances,
        "occurrence_assignments": occurrences,
        "dependency_edges": edges,
        "dependency_support": support,
        "manual_review": review,
    }


def check_partition_invariants(source_records, tables, sequence_set_id):
    """Independently audit interval coverage and graph components before completion.

    Reconstruction protects against checksum rewriting; these direct checks also
    prevent a construction regression from certifying its own malformed output.
    """
    records = tables["occurrence_assignments"]
    instances = tables["sequence_instances"]
    boundaries = tables["boundaries"]
    review = tables["manual_review"]
    edges = tables["dependency_edges"]
    if not records.frame_id.is_unique or not instances.sequence_id.is_unique:
        raise ValueError("Occurrence/sequence identities must be unique")
    pd.testing.assert_frame_equal(
        records[list(source_records)].sort_values("frame_id").reset_index(drop=True),
        source_records.sort_values("frame_id").reset_index(drop=True),
        check_exact=True,
    )
    if set(records.sequence_id) != set(instances.sequence_id):
        raise ValueError("Every sequence must have occurrences and vice versa")
    accepted = review.loc[review.decision.eq("accept")].sort_values(
        ["video_id", "localized_sample_index"]
    )
    if (
        boundaries.event_id.tolist() != accepted.event_id.tolist()
        or boundaries.localized_sample_index.tolist()
        != accepted.localized_sample_index.tolist()
        or boundaries.video_id.tolist() != accepted.video_id.tolist()
        or boundaries.boundary_type.tolist() != accepted.boundary_type.tolist()
    ):
        raise ValueError("Boundaries must equal exactly the sorted manual accepts")
    for video, rows in instances.groupby("video_id", sort=False):
        if (
            rows.start_sample_index.tolist()
            != [0, *(rows.end_sample_index.iloc[:-1] + 1).tolist()]
            or rows.end_sample_index.iloc[-1]
            != source_records.loc[
                source_records.video_id.eq(video), "sample_index"
            ].max()
            or rows.start_sample_index.iloc[1:].tolist()
            != boundaries.loc[
                boundaries.video_id.eq(video), "localized_sample_index"
            ].tolist()
        ):
            raise ValueError(
                "Sequence intervals/boundaries must cover each video without gaps or overlaps"
            )
    by_sequence = records.groupby("sequence_id", sort=False)
    for seq in instances.itertuples():
        occurrence = by_sequence.get_group(seq.sequence_id)
        if (
            len(occurrence) != seq.occurrence_count
            or seq.occurrence_count != seq.end_sample_index - seq.start_sample_index + 1
            or not occurrence.video_id.eq(seq.video_id).all()
            or not np.array_equal(
                occurrence.sample_index,
                np.arange(seq.start_sample_index, seq.end_sample_index + 1),
            )
            or seq.start_timestamp_seconds != occurrence.timestamp_seconds.iloc[0]
            or seq.end_timestamp_seconds != occurrence.timestamp_seconds.iloc[-1]
        ):
            raise ValueError("Invalid sequence occurrence coverage or timestamp bounds")
        expected_id = stable_id(
            {
                "kind": "sequence_instance",
                "sequence_set_id": sequence_set_id,
                "video_id": seq.video_id,
                "start_sample_index": seq.start_sample_index,
                "end_sample_index": seq.end_sample_index,
            }
        )
        if seq.sequence_id != expected_id:
            raise ValueError("Non-deterministic sequence ID")
    graph = {sid: set() for sid in instances.sequence_id}
    counts = records.groupby(["sequence_id", "content_id"]).size()
    for edge in edges.itertuples():
        left, right = edge.left_sequence_id, edge.right_sequence_id
        if (
            left == right
            or left not in graph
            or right not in graph
            or counts.get((left, edge.content_id), 0) != edge.left_occurrence_count
            or counts.get((right, edge.content_id), 0) != edge.right_occurrence_count
            or min(edge.left_occurrence_count, edge.right_occurrence_count) < 1
        ):
            raise ValueError("Dependency edge is not justified by exact shared content")
        graph[left].add(right)
        graph[right].add(left)
    groups = instances.set_index("sequence_id").exact_duplicate_dependency_group_id
    unseen = set(graph)
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(graph[node] - component)
        unseen -= component
        expected = stable_id(
            {
                "kind": "exact_duplicate_dependency_group",
                "sequence_set_id": sequence_set_id,
                "sequence_ids": sorted(component),
            }
        )
        if not groups.loc[sorted(component)].eq(expected).all():
            raise ValueError(
                "Dependency group IDs do not match recomputed connected components"
            )
    if (
        not records.exact_duplicate_dependency_group_id.eq(
            records.sequence_id.map(groups)
        ).all()
        or records.groupby("content_id")
        .exact_duplicate_dependency_group_id.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Exact duplicate occurrences cannot escape their dependency component"
        )
