"""Audit the completed single-floor verifier replay against the sparse labels."""

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCHMARK = REPO / "research/eval/benchmark_pairs.csv"
REVIEW = REPO / "research/experiments/20261019_upload/local/single_floor_comparison_20261017/人工复核标签_20261017.json"
OLD = REPO / "research/experiments/20261019_upload/local/single_floor_comparison_20261017/no_icp_summary.json"
SUMMARY = HERE / "comparison/no_icp_summary.json"
CANDIDATES = HERE / "comparison/all_candidates.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads(OLD.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert summary == old
    assert len(candidates) == summary["candidate_count"] == 1338

    positive = {}
    for row in csv.DictReader(BENCHMARK.open(encoding="utf-8-sig", newline="")):
        query = int(row["i_lap2"])
        if not (66 <= query <= 121 and row["label"] == "1" and row["uncertain"] == "0"):
            continue
        if float(row["t_i"]) - float(row["t_j"]) < 35:
            continue
        positive.setdefault(query, set()).add(int(row["j_lap1"]))
    assert len(positive) == 44

    by_query = defaultdict(list)
    for row in candidates:
        by_query[int(row["query"])].append(row)

    def hits(k):
        return sum(
            any(
                row["rank"] <= k and abs(row["match"] - partner) <= 1
                for row in by_query[query]
                for partner in partners
            )
            for query, partners in positive.items()
        )

    anchors = {(int(row["i"]), int(row["j"])) for row in summary["full_anchor_edges"]}
    reviewed_true = {tuple(row) for row in review["true_common_anchors"] + review["true_full_only_anchors"]}
    reviewed_false = {tuple(row) for row in review["false_loops"]}
    assert anchors == reviewed_true and not (anchors & reviewed_false)

    result = {
        "dataset": "20261008_145008",
        "candidate_count": len(candidates),
        "old_new_summary_identical": summary == old,
        "sha256": {"summary": sha(SUMMARY), "candidates": sha(CANDIDATES)},
        "stage_counts": {
            key: summary[key]
            for key in (
                "visual_pair_pass", "full_pair_pass", "visual_sequence_supported",
                "full_sequence_supported", "visual_spatial_anchors", "full_spatial_anchors"
            )
        },
        "sparse_benchmark": {
            "eligible_certain_positive_queries": len(positive),
            "retrieval_R_at_1": {"hits": hits(1), "denominator": len(positive)},
            "retrieval_R_at_5": {"hits": hits(5), "denominator": len(positive)},
            "note": "Query hit rate on sparse prior labels, not exhaustive true-loop recall.",
        },
        "reviewed_final_anchors": {"correct": len(anchors), "total": len(anchors)},
        "overall_precision": None,
        "overall_recall": None,
    }
    (HERE / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
