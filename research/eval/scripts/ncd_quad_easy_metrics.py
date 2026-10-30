#!/usr/bin/env python3
"""NCD Quad-Easy scoring (protocol v1-locked-20261023).

Implements:
  - R@k on canonical queries (default n=31 for adapter_GT_3s)
  - Recall on all GT positives (default n=38)
  - P_est / P_lb vs GT geometric positives (±1 kf match)
  - ATE SE(3) Umeyama (rigid, no scale) vs adapter GT nodes
  - start–end pose gap (ICP residual N/A without using submaps here)

Does NOT invent numbers: missing Vision/Backend inputs → status=missing, metrics TBD.

Front-end source column: default adapter_GT_3s; Omni-LIVO later as a separate column
(never mix node id spaces in one number).

Reuse patterns from eval/scripts/metrics.py (recall_at_k, loop_pr ±1 hit).
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
PROTOCOL_VERSION = 'v1-locked-20261023'
DEFAULT_GT = ROOT / 'ncd_quad_easy/gt_positives.csv'
DEFAULT_CANON = ROOT / 'ncd_quad_easy/gt_canonical_queries.csv'
DEFAULT_GT_POSES = PROJ / 'ncd_adapter/converted/quad-easy/keyframe_poses.txt'
DEFAULT_VISION = PROJ / 'vision/ncd_quad_easy'
DEFAULT_BACKEND = PROJ / 'backend/ncd_quad_easy'
DEFAULT_OUT = ROOT / 'ncd_quad_easy/results'


# ---------- I/O ----------

def load_poses_txt(path: Path):
    """t x y z qx qy qz qw → t, p (N,3), q (N,4 xyzw)."""
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def load_gt_positives(path: Path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r['j'] = int(r['j']); r['i'] = int(r['i'])
        r['d_m'] = float(r['d_m']); r['dt_s'] = float(r['dt_s'])
        r['dpsi_deg'] = float(r['dpsi_deg']); r['reverse'] = int(r['reverse'])
    return rows


def load_canonical(path: Path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r['query_i'] = int(r['query_i']); r['j_star'] = int(r['j_star'])
        r['d_m'] = float(r['d_m']); r['reverse'] = int(r['reverse'])
        r['gt_earliers'] = [int(x) for x in str(r.get('gt_earliers', '')).split() if x]
        if not r['gt_earliers']:
            r['gt_earliers'] = [r['j_star']]
    return rows


def edges_from(path: Path):
    if not path or not Path(path).exists():
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        out.append((int(parts[0]), int(parts[1])))
    return out


def load_candidates(path: Path):
    """Accept JSON list of {query_id|query|i, db_id|match|j|db, score} or CSV with those cols.
    Also accept dict with 'candidates' key. Returns list of (query, db, score)."""
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix.lower() == '.json':
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            raw = raw.get('candidates', raw.get('pairs', raw.get('results', [])))
        if not isinstance(raw, list):
            raise ValueError(f'candidates.json must be a list or dict-with-list, got {type(raw)}')
        out = []
        for c in raw:
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                out.append((int(c[0]), int(c[1]), float(c[2])))
                continue
            q = c.get('query_id', c.get('query', c.get('i', c.get('qi'))))
            d = c.get('db_id', c.get('match', c.get('candidate_id', c.get('j', c.get('db', c.get('dj'))))))
            s = c.get('score', c.get('sim', c.get('similarity', None)))
            if s is None and 'rank' in c:
                s = -float(c['rank'])  # lower rank = better
            if q is None or d is None or s is None:
                raise ValueError(f'candidate row missing fields: {c!r}')
            out.append((int(q), int(d), float(s)))
        return out
    # CSV
    rows = list(csv.DictReader(open(path)))
    out = []
    for c in rows:
        q = c.get('query_id', c.get('query', c.get('i')))
        d = c.get('db_id', c.get('match', c.get('candidate_id', c.get('j', c.get('db')))))
        s = c.get('score', c.get('sim'))
        out.append((int(q), int(d), float(s)))
    return out


# ---------- match helpers (metrics.py patterns) ----------

def _edge_hits_pair(e, j, i, tol=1):
    """Edge (a,b) matches ordered GT positive (j,i) within ±tol either orientation."""
    a, b = e
    return ((abs(a - i) <= tol and abs(b - j) <= tol) or
            (abs(b - i) <= tol and abs(a - j) <= tol))


def recall_at_k_ncd(cands, canonical, ks=(1, 5), tol=1):
    """R@k over canonical queries. Hit if any top-k db is within ±tol of ANY GT earlier.
    Queries with no candidates = miss. Denominator = len(canonical)."""
    by = defaultdict(list)
    for qi, dj, sc in (cands or []):
        by[qi].append((sc, dj))
    n = len(canonical)
    out = dict(n_queries=n,
               n_queries_with_candidates=sum(1 for r in canonical if r['query_i'] in by),
               tol=tol)
    for k in ks:
        h = 0
        for r in canonical:
            top = [d for _, d in sorted(by.get(r['query_i'], []), reverse=True)[:k]]
            earliers = r['gt_earliers']
            h += any(any(abs(d - j) <= tol for j in earliers) for d in top)
        out[f'recall@{k}'] = (h / n) if n else float('nan')
        out[f'hits@{k}'] = h
    return out


def loop_pr_ncd(edges, positives, tol=1):
    """P_est = TP/(TP+FP); P_lb = TP/|E|; Recall = covered_pos / |positives|.
    Edge is TP if it matches ≥1 GT positive (±tol); FP if it matches none
    (with GT available, unjudged→0: every unmatched edge is FP under geometric gate).
    Protocol: with GT, prefer geometric gate; unjudged only if we cannot decide —
    here every edge is decidable → unjudged=0, P_est==P_lb when all edges judged.
    Kept separate for paper-alignment with 20261008 naming."""
    if not positives:
        return dict(status='no GT positives', n_pos=0)
    tp = fp = unj = 0
    for e in edges:
        hits = [r for r in positives if _edge_hits_pair(e, r['j'], r['i'], tol)]
        if hits:
            tp += 1
        else:
            # With full GT geometric set, unmatched = FP (not unjudged)
            fp += 1
    cov = sum(any(_edge_hits_pair(e, r['j'], r['i'], tol) for e in edges) for r in positives)
    n_e = len(edges)
    n_pos = len(positives)
    return dict(
        n_pos=n_pos, n_edges=n_e, tp=tp, fp=fp, unjudged=unj,
        precision_est=(tp / (tp + fp)) if (tp + fp) else float('nan'),
        precision_lower_bound=(tp / n_e) if n_e else float('nan'),
        recall=(cov / n_pos) if n_pos else float('nan'),
        covered_pos=cov, tol=tol,
        n_reverse_pos=sum(r['reverse'] for r in positives),
        covered_reverse=sum(
            1 for r in positives if r['reverse'] and
            any(_edge_hits_pair(e, r['j'], r['i'], tol) for e in edges)),
    )


# ---------- ATE SE(3) Umeyama (no scale) ----------

def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """Rigid alignment src→dst (no scale). src,dst (N,3). Returns R(3,3), t(3,), rmse."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    S = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(S)
    Scorr = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Scorr[2, 2] = -1
    Rm = U @ Scorr @ Vt
    t = mu_d - Rm @ mu_s
    aligned = (Rm @ src.T).T + t
    err = np.linalg.norm(aligned - dst, axis=1)
    return Rm, t, aligned, err


def ate_se3(est_p: np.ndarray, gt_p: np.ndarray):
    Rm, t, aligned, err = umeyama_se3(est_p, gt_p)
    return dict(
        method='SE3_Umeyama_no_scale',
        n=int(len(err)),
        ate_rmse_m=float(np.sqrt(np.mean(err ** 2))),
        ate_trans_median_m=float(np.median(err)),
        ate_trans_mean_m=float(np.mean(err)),
        ate_z_rmse_m=float(np.sqrt(np.mean((aligned[:, 2] - gt_p[:, 2]) ** 2))),
        ate_xy_rmse_m=float(np.sqrt(np.mean(np.sum((aligned[:, :2] - gt_p[:, :2]) ** 2, axis=1)))),
    )


def start_end_gap(p: np.ndarray):
    g = p[-1] - p[0]
    return dict(
        pose_gap_m=float(np.linalg.norm(g)),
        pose_gap_xy_m=float(np.linalg.norm(g[:2])),
        pose_gap_z_m=float(g[2]),
        icp_residual_m='N/A',  # protocol: no submap ICP in this scorer path
    )


# ---------- main run ----------

def resolve_vision_candidates(vision_dir: Path):
    for name in ('candidates.json', 'candidates.csv'):
        p = vision_dir / name
        if p.exists():
            return p
    return None


def resolve_backend(backend_dir: Path):
    return dict(
        edges=backend_dir / 'edges_accepted.edges',
        poses_opt=backend_dir / 'poses_optimised.txt',
        poses_odom=backend_dir / 'poses_odometry.txt',
        edges_verified=None,  # optional under vision
    )


def run(args):
    positives = load_gt_positives(args.gt)
    canonical = load_canonical(args.canonical)
    gt_t, gt_p, gt_q = load_poses_txt(args.gt_poses)

    vision_cand_path = resolve_vision_candidates(args.vision_dir)
    backend = resolve_backend(args.backend_dir)
    if args.candidates:
        vision_cand_path = Path(args.candidates)
    if args.edges:
        backend['edges'] = Path(args.edges)
    if args.poses_opt:
        backend['poses_opt'] = Path(args.poses_opt)
    if args.poses_odom:
        backend['poses_odom'] = Path(args.poses_odom)

    missing = []
    res = dict(
        protocol_version=PROTOCOL_VERSION,
        name=args.name,
        node_source=args.node_source,
        frontend=args.frontend,
        thr_m=args.thr_m,
        dt_min_s=args.dt_min_s,
        n_nodes_gt=int(len(gt_p)),
        n_gt_positives=len(positives),
        n_canonical_queries=len(canonical),
        n_reverse_positives=sum(r['reverse'] for r in positives),
        gt_start_end=start_end_gap(gt_p),
        inputs={},
        missing=[],
    )

    # --- R@k ---
    if vision_cand_path and vision_cand_path.exists():
        cands = load_candidates(vision_cand_path)
        res['inputs']['candidates'] = str(vision_cand_path)
        res['R_at_k'] = recall_at_k_ncd(cands, canonical, ks=tuple(args.ks), tol=args.tol)
        res['R_at_k']['status'] = 'ok'
    else:
        missing.append(str(args.vision_dir / 'candidates.json'))
        res['R_at_k'] = dict(status='missing', recall_at_1='TBD', recall_at_5='TBD',
                             n_queries=len(canonical), note='Vision candidates.json not found')

    # --- edges / P/R ---
    edges_path = Path(backend['edges'])
    if edges_path.exists():
        edges = edges_from(edges_path)
        res['inputs']['edges_accepted'] = str(edges_path)
        res['n_accepted_edges'] = len(edges)
        res['loop_pr'] = loop_pr_ncd(edges, positives, tol=args.tol)
        res['loop_pr']['status'] = 'ok'
        # aliases matching protocol names
        res['loop_pr']['P_est'] = res['loop_pr']['precision_est']
        res['loop_pr']['P_lb'] = res['loop_pr']['precision_lower_bound']
    else:
        missing.append(str(edges_path))
        res['n_accepted_edges'] = 'TBD'
        res['loop_pr'] = dict(status='missing', P_est='TBD', P_lb='TBD', recall='TBD',
                              n_pos=len(positives), note='Backend edges_accepted.edges not found')

    # optional verified edges under vision
    ver = args.vision_dir / 'edges_verified.edges'
    if ver.exists():
        res['inputs']['edges_verified'] = str(ver)
        res['n_verified_edges'] = len(edges_from(ver))
    else:
        res['n_verified_edges'] = 'TBD (optional)'

    # --- ATE / start-end on optimised + odometry ---
    for key, label in (('poses_opt', 'optimised'), ('poses_odom', 'odometry')):
        path = Path(backend[key])
        slot = f'ate_{label}'
        if path.exists():
            _, ep, _ = load_poses_txt(path)
            if len(ep) != len(gt_p):
                res[slot] = dict(status='error',
                                 note=f'pose count {len(ep)} != GT {len(gt_p)}', path=str(path))
            else:
                res['inputs'][key] = str(path)
                res[slot] = ate_se3(ep, gt_p)
                res[slot]['status'] = 'ok'
                res[slot]['start_end'] = start_end_gap(ep)
                res[slot]['path'] = str(path)
        else:
            missing.append(str(path))
            res[slot] = dict(status='missing', ate_rmse_m='TBD', path=str(path))

    res['map_metrics'] = dict(floor_z='N/A', lap_p2pl='N/A', wall_sharpness='N/A',
                              note='protocol §3.5')
    res['missing'] = missing
    res['ready_to_publish'] = (len(missing) == 0)
    return res


def write_skeleton_tables(out_dir: Path, res: dict):
    """CSV + MD result tables with TBD where missing."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell(block, *keys, default='TBD'):
        if not isinstance(block, dict):
            return default
        if block.get('status') == 'missing':
            # try nested TBD already set
            cur = block
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur if cur is not None else default
        cur = block
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        if isinstance(cur, float):
            return f'{cur:.6f}' if abs(cur) < 1e3 else cur
        return cur

    rak = res.get('R_at_k', {})
    pr = res.get('loop_pr', {})
    ate_o = res.get('ate_optimised', {})
    ate_r = res.get('ate_odometry', {})

    rows = [
        ('frontend', res.get('frontend')),
        ('node_source', res.get('node_source')),
        ('protocol_version', res.get('protocol_version')),
        ('thr_m', res.get('thr_m')),
        ('dt_min_s', res.get('dt_min_s')),
        ('n_gt_positives', res.get('n_gt_positives')),
        ('n_canonical_queries', res.get('n_canonical_queries')),
        ('n_reverse_positives', res.get('n_reverse_positives')),
        ('R@1', cell(rak, 'recall@1')),
        ('R@5', cell(rak, 'recall@5')),
        ('n_accepted_edges', res.get('n_accepted_edges', 'TBD')),
        ('P_est', cell(pr, 'P_est') if 'P_est' in pr or pr.get('status') == 'ok' else cell(pr, 'precision_est')),
        ('P_lb', cell(pr, 'P_lb') if 'P_lb' in pr or pr.get('status') == 'ok' else cell(pr, 'precision_lower_bound')),
        ('Recall', cell(pr, 'recall')),
        ('ATE_RMSE_SE3_optimised_m', cell(ate_o, 'ate_rmse_m')),
        ('ATE_trans_median_optimised_m', cell(ate_o, 'ate_trans_median_m')),
        ('ATE_Z_RMSE_optimised_m', cell(ate_o, 'ate_z_rmse_m')),
        ('start_end_gap_optimised_m', cell(ate_o, 'start_end', 'pose_gap_m') if isinstance(ate_o.get('start_end'), dict) else 'TBD'),
        ('ATE_RMSE_SE3_odometry_m', cell(ate_r, 'ate_rmse_m')),
        ('start_end_gap_odometry_m', cell(ate_r, 'start_end', 'pose_gap_m') if isinstance(ate_r.get('start_end'), dict) else 'TBD'),
        ('start_end_gap_GT_m', f"{res['gt_start_end']['pose_gap_m']:.6f}"),
        ('floor_z / lap_p2pl / sharpness', 'N/A'),
        ('ready_to_publish', res.get('ready_to_publish')),
        ('missing_artifacts', '; '.join(res.get('missing') or []) or '(none)'),
    ]

    csv_path = out_dir / f'metrics_{res["name"]}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerows(rows)

    md_path = out_dir / 'table.md'
    lines = [
        f'# NCD Quad-Easy metrics — `{res["name"]}`',
        '',
        f'- **protocol:** `{res.get("protocol_version")}`',
        f'- **frontend:** `{res.get("frontend")}` (do not mix with Omni-LIVO in one number)',
        f'- **GT gate:** `< {res.get("thr_m")} m` 3D and `dt > {res.get("dt_min_s")} s`',
        f'- **denominators:** R@k = canonical **{res.get("n_canonical_queries")}**; Recall = all **{res.get("n_gt_positives")}** positives (incl. {res.get("n_reverse_positives")} reverse)',
        f'- **external positioning:** same-direction external validation; reverse-loop claim stays on `20261008`',
        f'- **ready_to_publish:** {res.get("ready_to_publish")}',
        '',
        '| metric | value |',
        '|---|---|',
    ]
    for k, v in rows:
        lines.append(f'| {k} | {v} |')
    if res.get('missing'):
        lines += ['', '## Missing artifacts (blocked)', '']
        for m in res['missing']:
            lines.append(f'- `{m}`')
        lines += ['', 'Vision should write `candidates.json` (and optional `edges_verified.edges`) under '
                  '`project/vision/ncd_quad_easy/`. Backend should write `edges_accepted.edges`, '
                  '`poses_optimised.txt`, `poses_odometry.txt` under `project/backend/ncd_quad_easy/`.']
    md_path.write_text('\n'.join(lines) + '\n')

    # also a wide one-row CSV for paper paste
    wide_path = out_dir / 'table.csv'
    headers = [k for k, _ in rows]
    values = [v for _, v in rows]
    with open(wide_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(headers); w.writerow(values)

    return csv_path, md_path, wide_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='ncd_quad_easy_adapter3s')
    ap.add_argument('--gt', type=Path, default=DEFAULT_GT)
    ap.add_argument('--canonical', type=Path, default=DEFAULT_CANON)
    ap.add_argument('--gt-poses', type=Path, default=DEFAULT_GT_POSES)
    ap.add_argument('--vision-dir', type=Path, default=DEFAULT_VISION)
    ap.add_argument('--backend-dir', type=Path, default=DEFAULT_BACKEND)
    ap.add_argument('--candidates', type=Path, default=None, help='override Vision candidates path')
    ap.add_argument('--edges', type=Path, default=None)
    ap.add_argument('--poses-opt', type=Path, default=None)
    ap.add_argument('--poses-odom', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--node-source', default='adapter_3s')
    ap.add_argument('--frontend', default='adapter_GT_3s')
    ap.add_argument('--thr-m', type=float, default=2.0)
    ap.add_argument('--dt-min-s', type=float, default=30.0)
    ap.add_argument('--tol', type=int, default=1)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 5])
    ap.add_argument('--score-fake', action='store_true',
                    help='DEV ONLY: also accept backend fake_candidates.json (never for publish)')
    a = ap.parse_args()

    if not a.gt.exists() or not a.canonical.exists():
        print('GT positives missing; run build_ncd_gt_positives.py first', file=sys.stderr)
        return 2

    # refuse to silently score fake/smoke as publishable
    if a.score_fake and a.candidates is None:
        fake = a.backend_dir / 'fake_candidates.json'
        if fake.exists():
            a.candidates = fake
            a.name = a.name + '_FAKE_DEV'

    res = run(a)
    a.out.mkdir(parents=True, exist_ok=True)
    json_path = a.out / f'metrics_{a.name}.json'
    json.dump(res, open(json_path, 'w'), indent=2, default=str)
    csv_path, md_path, wide_path = write_skeleton_tables(a.out, res)
    print(json_path)
    print(csv_path)
    print(md_path)
    print(wide_path)
    print('ready_to_publish=', res['ready_to_publish'], 'missing=', len(res['missing']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
