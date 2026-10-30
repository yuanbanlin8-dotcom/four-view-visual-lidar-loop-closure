"""LiDAR-only ICP checks for Scan Context candidates on raw Omni-LIVO poses.

The post-hoc floor labels are not used to accept or reject a candidate.
Results are checkpointed after each pair for a recoverable CPU run.
"""

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
SUBMAPS = ROOT.parent / (ROOT.name + '_submaps')
OUT = ROOT / 'comparison_20261016/sc_raw_omnilivo_20261017'
sys.path.insert(0, str(ROOT / 'loop_latest_20261014/source_snapshot/eval/scripts'))
os.environ['SEEKER_SUBMAP_ROOT'] = str(SUBMAPS)
from common import T_of, icp_p2pl, normals, transform, voxel  # noqa: E402


@lru_cache(maxsize=32)
def submap(i):
    return np.load(SUBMAPS / f'{i:04d}.npy').astype(np.float64)


def check_pair(pair, poses):
    i, j = pair['i'], pair['j']
    raw_distance = float(np.linalg.norm(poses[i, 1:4] - poses[j, 1:4]))
    row = {'i': i, 'j': j, 'sc_distance': pair['sc_distance'],
           'raw_position_distance_m': raw_distance}
    # This generous odometry-consistency precheck is applied equally to all
    # candidates. It avoids expensive ICP for pairs with no plausible overlap.
    if raw_distance > 15.0:
        row['status'] = 'raw_pose_distance_over_15m'
        return row
    ti = T_of(poses[i, 1:4], poses[i, 4:8])
    tj = T_of(poses[j, 1:4], poses[j, 4:8])
    initial = np.linalg.inv(tj) @ ti
    t0 = time.monotonic()
    refined, fitness, rmse, count, eigenvalues = icp_p2pl(
        submap(i), submap(j), initial, iters=30)
    correction = np.linalg.inv(initial) @ refined
    correction_deg = float(np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude()))
    finite_eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    row.update(forward_fitness=fitness, forward_rmse_m=rmse if np.isfinite(rmse) else None,
               forward_count=count, correction_translation_m=float(np.linalg.norm(correction[:3, 3])),
               correction_rotation_deg=correction_deg,
               translational_hessian_min=(float(np.min(finite_eigenvalues))
                                          if len(finite_eigenvalues) else None),
               elapsed_seconds=time.monotonic() - t0)
    if fitness < .25 or not np.isfinite(rmse) or rmse > .18:
        row['status'] = 'forward_icp_reject'
        return row
    source, target = voxel(submap(i), .2), voxel(submap(j), .2)
    source_normals, source_planarity, tree = normals(source)
    reverse = transform(np.linalg.inv(refined), target)
    distances, indices = tree.query(reverse, distance_upper_bound=.3)
    valid = np.isfinite(distances)
    valid[valid] &= source_planarity[indices[valid]] > .3
    reverse_rmse = (float(np.sqrt(np.mean(np.sum(
        (reverse[valid] - source[indices[valid]]) * source_normals[indices[valid]], axis=1) ** 2)))
        if valid.any() else None)
    row.update(reverse_fitness=float(np.mean(valid)), reverse_rmse_m=reverse_rmse)
    if (row['reverse_fitness'] < .25 or reverse_rmse is None or reverse_rmse > .18):
        row['status'] = 'reverse_icp_reject'
    elif row['correction_translation_m'] > 3.0 or correction_deg > 20.0:
        row['status'] = 'large_correction_reject'
    else:
        row['status'] = 'geometry_pass'
        row['relative_transform'] = refined.tolist()
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = json.loads((OUT / 'retrieval_summary.json').read_text())['reports']
    chosen = next(x for x in reports if x['gap_nodes'] == 8
                  and x['ring_key_shortlist'] == 3 and x['sc_threshold'] == .20)
    poses = np.loadtxt(ROOT / 'loop_latest_20261014/raw_poses.txt')
    path = OUT / 'geometry_gap8_k3_thresh020.jsonl'
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[(entry['i'], entry['j'])] = entry
    with path.open('a') as f:
        for pair in chosen['pairs']:
            key = pair['i'], pair['j']
            if key in done:
                continue
            result = check_pair(pair, poses)
            f.write(json.dumps(result, allow_nan=False) + '\n')
            f.flush()
            print(len(done) + 1, len(chosen['pairs']), result['i'], result['j'],
                  result['status'], round(result.get('elapsed_seconds', 0), 2), flush=True)
            done[key] = result


if __name__ == '__main__':
    main()
