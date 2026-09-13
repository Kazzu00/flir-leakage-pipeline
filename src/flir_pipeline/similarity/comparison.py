"""Neighbor-set agreement across separate feature spaces, without combining them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flir_pipeline.similarity.cosine import distribution_summary
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    verify_similarity_directory,
    write_json,
)


def compare_neighbor_spaces(left: pd.DataFrame, right: pd.DataFrame,
                            ks: tuple[int, ...] = (1, 5, 10, 20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align query IDs explicitly and compute Jaccard |A∩B|/|A∪B| for each k.

    k=1 Jaccard is exactly the nearest-neighbor match indicator. Missing queries,
    incomplete ranks, duplicate neighbors and self-neighbors are errors, not zeros.
    """
    if not ks or len(set(ks)) != len(ks) or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("ks must be distinct positive integers")
    if set(left.query_content_id) != set(right.query_content_id) or left.empty:
        raise ValueError("Neighbor spaces must cover the same nonempty query set")
    grouped = []
    for frame in (left, right):
        groups = {}
        for query, group in frame.groupby("query_content_id", sort=True):
            selected = group.sort_values("neighbor_rank").loc[lambda x: x.neighbor_rank <= max(ks)]
            if selected.neighbor_rank.tolist() != list(range(1, max(ks)+1)) or not selected.neighbor_content_id.is_unique or selected.neighbor_content_id.eq(query).any():
                raise ValueError("Incomplete or invalid neighbor rankings")
            groups[query] = selected.neighbor_content_id.tolist()
        grouped.append(groups)
    rows = []
    for query in sorted(grouped[0]):
        for k in ks:
            a, b = set(grouped[0][query][:k]), set(grouped[1][query][:k])
            rows.append({"query_content_id": query, "k": k, "intersection_count": len(a & b),
                         "union_count": len(a | b), "jaccard": len(a & b)/len(a | b)})
    contents = pd.DataFrame(rows)
    summaries = []
    for k in ks:
        values = contents.loc[contents.k == k, "jaccard"].to_numpy()
        stats = distribution_summary(values)
        summaries.append({"k": k, "contents": stats["count"], "mean_jaccard": stats["mean"],
                          "median_jaccard": stats["median"], "Q1": stats["Q1"], "Q3": stats["Q3"],
                          "exact_neighbor_match_percentage": 100*stats["mean"] if k == 1 else None})
    return contents, pd.DataFrame(summaries)


def compare_to_store(left: Path, right: Path, output_root: Path = Path("artifacts/similarity/comparisons"),
                     ks: tuple[int, ...] = (1, 5, 10, 20)) -> Path:
    for directory in (left, right):
        if not verify_similarity_directory(directory)["quality_valid"]:
            raise ValueError("Comparison requires verified similarity artifacts")
    a, b = read_json(left/"metadata.json"), read_json(right/"metadata.json")
    if a["dataset_id"] != b["dataset_id"]:
        raise ValueError("Cannot compare different datasets")
    if a["feature_space_id"] == b["feature_space_id"]:
        raise ValueError("Select two different feature spaces")
    signature = {"dataset_id": a["dataset_id"], "left_similarity_space_id": a["similarity_space_id"],
                 "right_similarity_space_id": b["similarity_space_id"], "ks": list(ks), "metric": "neighbor_set_jaccard_v1",
                 "left_neighbors_sha256": file_sha256(left/"nearest_neighbors.parquet"),
                 "right_neighbors_sha256": file_sha256(right/"nearest_neighbors.parquet")}
    comparison_id = stable_id(signature)
    output = output_root/a["dataset_id"]/comparison_id
    if output.exists():
        if not (output/"metadata.json").is_file():
            raise ValueError("Incomplete comparison preserved; select a separate output root")
        meta = read_json(output/"metadata.json")
        required = {"neighbor_agreement.parquet", "neighbor_agreement_summary.csv"}
        if meta.get("signature") != signature or meta.get("comparison_id") != comparison_id or set(meta.get("output_sha256", {})) != required or not all(file_sha256(output/name) == meta["output_sha256"][name] for name in required):
            raise ValueError("Comparison cache failed integrity verification")
        return output
    contents, summary = compare_neighbor_spaces(pd.read_parquet(left/"nearest_neighbors.parquet"), pd.read_parquet(right/"nearest_neighbors.parquet"), ks)
    output.mkdir(parents=True, exist_ok=False)
    contents.to_parquet(output/"neighbor_agreement.parquet", index=False)
    summary.to_csv(output/"neighbor_agreement_summary.csv", index=False)
    write_json(output/"metadata.json", {**execution_provenance(), "comparison_id": comparison_id, "signature": signature,
                                       "output_sha256": {name: file_sha256(output/name) for name in ("neighbor_agreement.parquet", "neighbor_agreement_summary.csv")}})
    return output
