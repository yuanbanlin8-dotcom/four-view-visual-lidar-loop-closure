"""Re-evaluate frozen 2026-10-08 single-floor outputs against existing labels.

This evaluates cached outputs from the frozen run. It does not retrain descriptors or
rerun SuperPoint, PnP, or ICP. The benchmark is sparse, so full-set precision and
recall cannot be inferred from it.
"""

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
UPLOAD = REPO / "research"
BENCHMARK = UPLOAD / "eval" / "benchmark_pairs.csv"
SHORTLIST = UPLOAD / "experiments" / "20261019_upload" / "nuc" / "single_floor" / "no_icp" / "shortlist.json"
SUMMARY = HERE / "no_icp_summary.json"
REVIEW = HERE / "人工复核标签_20261017.json"
OUT = HERE / "frozen_single_floor_metrics_20261020.json"
MD = HERE / "frozen_single_floor_metrics_20261020.md"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hits(rows, positives, k):
    return sum(any(r["rank"] <= k and abs(r["match"] - p) <= 1 for r in rows[q] for p in positives[q]) for q in positives)


def main():
    bench = list(csv.DictReader(BENCHMARK.open(encoding="utf-8-sig", newline="")))
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert len(shortlist) == summary["candidate_count"] == 1338
    assert len(summary["full_anchor_edges"]) == 17

    rows = defaultdict(list)
    for r in shortlist:
        rows[int(r["query"])].append(r)
    for q, ranked in rows.items():
        assert sorted(r["rank"] for r in ranked) == list(range(1, len(ranked) + 1)), q

    positive = defaultdict(set)
    uncertain = set()
    excluded_short_gap = set()
    for r in bench:
        q, j = int(r["i_lap2"]), int(r["j_lap1"])
        if not (66 <= q <= 121 and r["label"] == "1"):
            continue
        if float(r["t_i"]) - float(r["t_j"]) < 35:
            excluded_short_gap.add(q)
            continue
        positive[q].add(j)
        if r["uncertain"] == "1":
            uncertain.add(q)
    assert len(positive) == 49
    strict = {q: js for q, js in positive.items() if q not in uncertain}
    assert all(q in rows for q in strict)

    anchor_rows = defaultdict(list)
    for e in summary["full_anchor_edges"]:
        anchor_rows[int(e["i"])].append({"rank": 1, "match": int(e["j"])})
    reviewed_true = {tuple(x) for x in review["true_common_anchors"] + review["true_full_only_anchors"]}
    reviewed_false = {tuple(x) for x in review["false_loops"]}
    anchors = {(int(e["i"]), int(e["j"])) for e in summary["full_anchor_edges"]}
    assert anchors == reviewed_true and not anchors & reviewed_false

    result = {
        "dataset": "20261008_145008",
        "evaluation": "frozen retrieval rerun on NUC 2026-10-20, metrics scored locally",
        "nuc_rerun_check": {
            "host": "192.168.10.15",
            "script": "run_single_floor_retrieval.py",
            "source_descriptors": "/home/nuc/omnilivo_floor_loop_trial/single_floor_20261008_no_icp_20261017/vision_input/global_dinov2.npz",
            "rerun_candidate_count": 1338,
            "same_query_match_order": True,
            "same_rank_rows": 1338,
            "same_score_rows": 91,
            "rerun_shortlist_sha256": "c10d4f1b5efcaab062488bd9ac003cc20ff0b963f1d2935f20ad6797e4113641",
            "note": "Floating-point score values changed slightly; selected pairs and ranks did not.",
        },
        "source_sha256": {p.relative_to(REPO).as_posix(): sha256(p) for p in (BENCHMARK, SHORTLIST, SUMMARY, REVIEW)},
        "query_scope": "lap-2 carried nodes 66..121",
        "match_tolerance_nodes": 1,
        "retrieval_min_time_gap_s": 35,
        "all_human_positive_queries": 52,
        "excluded_positive_queries_below_35s": sorted(excluded_short_gap),
        "eligible_positive_queries_including_uncertain": len(positive),
        "uncertain_eligible_queries": sorted(uncertain),
        "strict_positive_queries": len(strict),
        "retrieval_R_at_1": {"hits": hits(rows, strict, 1), "denominator": len(strict)},
        "retrieval_R_at_5": {"hits": hits(rows, strict, 5), "denominator": len(strict)},
        "retrieval_R_at_1_including_uncertain": {"hits": hits(rows, positive, 1), "denominator": len(positive)},
        "retrieval_R_at_5_including_uncertain": {"hits": hits(rows, positive, 5), "denominator": len(positive)},
        "accepted_final_anchors": len(anchors),
        "final_anchor_query_coverage_of_strict_benchmark": {"hits": hits(anchor_rows, strict, 1), "denominator": len(strict)},
        "reviewed_final_anchor_correctness": {"correct": len(anchors & reviewed_true), "reviewed": len(anchors)},
        "full_precision": None,
        "full_recall": None,
        "limitation": "Benchmark positives are sparse and some labels derive from odometry proximity; negatives are auto-generated. Human review covers selected final anchors only. No exhaustive independent pair-level ground truth exists.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r1, r5 = result["retrieval_R_at_1"], result["retrieval_R_at_5"]
    cov = result["final_anchor_query_coverage_of_strict_benchmark"]
    MD.write_text(
        "# 冻结单层流程的基准重算（2026-10-20）\n\n"
        "数据集 `20261008_145008`；在 NUC 上使用冻结的 DINO 描述子重跑检索，得到 1338 个候选，"
        "点对及排名与归档输出逐行一致，只有浮点分数存在微小变化。"
        "复用此前完整验证形成的 17 个最终锚点重新计算指标；未重新提取特征或执行 SuperPoint、PnP、ICP。"
        "评估去程/回程中回程的 66–121 号节点；匹配允许旧基准使用的 ±1 节点容差。\n\n"
        f"旧基准在该范围有 52 个正例查询。其中 {len(excluded_short_gap)} 个正例的时间差小于现行检索的 35 s 排除阈值，"
        f"不能作为当前流程的可检出正例；另外 {len(uncertain)} 个可检正例原本带有人工不确定标记。"
        f"严格口径剩 {len(strict)} 个正例查询。\n\n"
        "| 指标 | 严格口径 | 含不确定标签 |\n|---|---:|---:|\n"
        f"| 检索 R@1 | {r1['hits']}/{r1['denominator']} = {r1['hits']/r1['denominator']:.1%} | "
        f"{result['retrieval_R_at_1_including_uncertain']['hits']}/{len(positive)} |\n"
        f"| 检索 R@5 | {r5['hits']}/{r5['denominator']} = {r5['hits']/r5['denominator']:.1%} | "
        f"{result['retrieval_R_at_5_including_uncertain']['hits']}/{len(positive)} |\n"
        f"| 最终锚点覆盖的正例查询 | {cov['hits']}/{cov['denominator']} | — |\n\n"
        "这里的 R@k 是**稀疏基准的查询命中率**，不是全数据集所有真实回环的召回率。"
        "17 个最终锚点中，人工复核的 17 个被判为正确；这是已复核锚点的 17/17，不能写作总体 precision=100%。"
        "由于未对全部可能点对独立穷尽标注，整体 precision 和 recall 均不可可靠计算。"
        "旧基准的负例是自动生成的，正例部分由里程计邻近性选取；上述数字应标明基准限制。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
