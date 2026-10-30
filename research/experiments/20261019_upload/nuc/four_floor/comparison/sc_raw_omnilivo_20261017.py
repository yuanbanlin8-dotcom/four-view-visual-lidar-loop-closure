"""Scan Context retrieval on the frozen, unoptimized Omni-LIVO submaps.

Reimplements the 20x60 max-height polar context, ring/sector keys, three
ring-key neighbours and seven-shift cosine comparison of the SC-PGO copy on
the NUC. No visual data, corrected poses, floor labels or loop edges enter
retrieval. Floor labels and accepted edges are read only for post-hoc audit.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
SUBMAPS = ROOT.parent / (ROOT.name + '_submaps')
OUT = ROOT / 'comparison_20261016' / 'sc_raw_omnilivo_20261017'
RINGS, SECTORS, RADIUS = 20, 60, 80.0


def make_context(points):
    xyz = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(xyz[:, :2], axis=1)
    angle = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    keep = np.isfinite(xyz).all(axis=1) & (r <= RADIUS)
    r, angle, height = r[keep], angle[keep], xyz[keep, 2] + 2.0
    ring = np.maximum(1, np.minimum(RINGS, np.ceil(r * RINGS / RADIUS).astype(int))) - 1
    sector = np.maximum(1, np.minimum(SECTORS, np.ceil(angle * SECTORS / 360.0).astype(int))) - 1
    desc = np.full((RINGS, SECTORS), -1000.0)
    np.maximum.at(desc, (ring, sector), height)
    desc[desc == -1000.0] = 0.0
    return desc


def context_distance(a, b, sector_a, sector_b):
    shifts = np.array([np.linalg.norm(sector_a - np.roll(sector_b, s)) for s in range(SECTORS)])
    center = int(np.argmin(shifts))
    best = (float('inf'), 0)
    for offset in (0, 1, -1, 2, -2, 3, -3):
        shift = (center + offset) % SECTORS
        rolled = np.roll(b, shift, axis=1)
        norms = np.linalg.norm(a, axis=0) * np.linalg.norm(rolled, axis=0)
        good = norms > 0
        if not np.any(good):
            continue
        cosine = np.sum(a[:, good] * rolled[:, good], axis=0) / norms[good]
        distance = float(1.0 - np.mean(cosine))
        if distance < best[0]:
            best = (distance, shift)
    return best


def candidates(desc, ring_keys, sector_keys, gap, shortlist=10):
    out = []
    for i in range(len(desc)):
        eligible = i - gap
        if eligible <= 0:
            continue
        k = min(shortlist, eligible)
        distances, indices = cKDTree(ring_keys[:eligible]).query(ring_keys[i], k=k)
        indices = np.atleast_1d(indices).astype(int)
        distances = np.atleast_1d(distances)
        for rank, (j, ring_distance) in enumerate(zip(indices, distances), start=1):
            sc_distance, shift = context_distance(desc[i], desc[j], sector_keys[i], sector_keys[j])
            out.append({'i': i, 'j': int(j), 'ring_rank': rank,
                        'ring_key_distance': float(ring_distance),
                        'sc_distance': sc_distance, 'yaw_shift_deg': int(shift * 6)})
    return out


def summarize(rows, gap, shortlist, threshold, labels, accepted):
    by_query = {}
    for row in rows:
        if row['ring_rank'] <= shortlist:
            current = by_query.get(row['i'])
            if current is None or row['sc_distance'] < current['sc_distance']:
                by_query[row['i']] = row
    found = [x for x in by_query.values() if x['sc_distance'] < threshold]
    same = [x for x in found if labels[x['i']] is not None
            and labels[x['j']] is not None and labels[x['i']] == labels[x['j']]]
    cross = [x for x in found if labels[x['i']] is not None
             and labels[x['j']] is not None and labels[x['i']] != labels[x['j']]]
    unknown = [x for x in found if labels[x['i']] is None or labels[x['j']] is None]
    covered = sum(any(abs(x['i'] - e['i']) <= 2 and abs(x['j'] - e['j']) <= 2
                      for x in found) for e in accepted)
    return {'gap_nodes': gap, 'ring_key_shortlist': shortlist, 'sc_threshold': threshold,
            'queries_with_candidate': len(by_query), 'top1_below_threshold': len(found),
            'same_floor_posthoc': len(same), 'cross_floor_posthoc': len(cross),
            'unknown_floor_posthoc': len(unknown),
            'accepted_edge_neighborhoods_covered_of_26': covered,
            'pairs': [{'i': x['i'], 'j': x['j'], 'sc_distance': x['sc_distance'],
                       'floor_relation_posthoc': ('unknown' if labels[x['i']] is None or labels[x['j']] is None
                                                 else 'same' if labels[x['i']] == labels[x['j']] else 'cross')}
                      for x in sorted(found, key=lambda x: (x['i'], x['j']))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pose_rows = json.loads((SUBMAPS / 'poses.json').read_text())
    raw = np.loadtxt(ROOT / 'loop_latest_20261014/raw_poses.txt')
    assert len(pose_rows) == len(raw) == 243
    assert max(abs(pose_rows[i]['t'] - raw[i, 0]) for i in range(243)) < 1e-6
    assert max(np.linalg.norm(np.asarray(pose_rows[i]['p']) - raw[i, 1:4]) for i in range(243)) < 1e-6
    desc = [make_context(np.load(SUBMAPS / f'{i:04d}.npy')) for i in range(len(raw))]
    ring_keys = np.stack([x.mean(axis=1) for x in desc])
    sector_keys = [x.mean(axis=0) for x in desc]
    floor = json.loads((ROOT / 'loop_latest_20261014/floor_hypotheses.json').read_text())
    labels = floor['labels']
    accepted = json.loads((ROOT / 'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())['accepted_edges']
    reports = []
    for gap in (8, 30):
        rows = candidates(desc, ring_keys, sector_keys, gap)
        with (OUT / f'candidates_gap{gap}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for shortlist in (3, 10):
            for threshold in (0.13, 0.20):
                reports.append(summarize(rows, gap, shortlist, threshold, labels, accepted))
    result = {'source_submaps': str(SUBMAPS),
              'source_poses': str(ROOT / 'loop_latest_20261014/raw_poses.txt'),
              'submap_count': 243,
              'algorithm': '20x60 max-height Scan Context; 80 m radius; +2 m height; ring-key KNN; sector-key yaw alignment +/-3 sectors; cosine distance',
              'floors_used_for_retrieval': False, 'corrected_poses_used': False,
              'elapsed_seconds': time.monotonic() - started,
              'reports': reports}
    (OUT / 'retrieval_summary.json').write_text(json.dumps(result, indent=2))
    for x in reports:
        print({k: v for k, v in x.items() if k != 'pairs'}, flush=True)


if __name__ == '__main__':
    main()
