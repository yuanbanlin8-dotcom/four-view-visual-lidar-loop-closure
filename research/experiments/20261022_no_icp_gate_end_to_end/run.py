"""Finish the frozen four-floor no-symmetric-ICP-gate ablation."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


BASE = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
RELEASE = BASE / 'loop_latest_20261014'
SUBMAPS = BASE.parent / 'four_floor_20261013_160929_submaps'
REPO = Path('/home/nuc/five_floor_optimization_20261013')
OUT = BASE / 'no_icp_gate_end_to_end_20261022'
OUT.mkdir(parents=True, exist_ok=True)

os.environ.update(
    SEEKER_RESEARCH_ROOT=str(REPO / 'research'),
    SEEKER_SUBMAP_ROOT=str(SUBMAPS),
    SEEKER_RUNTIME_CONFIG=str(BASE / 'loop_inputs/runtime.yaml'),
    SEEKER_CAMERA_CONFIG=str(REPO / 'mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'),
    SEEKER_VISION_ROOT=str(RELEASE / 'vision'),
    SEEKER_MASK='v2',
    OPENBLAS_NUM_THREADS='4',
    OMP_NUM_THREADS='4',
)
sys.path[:0] = [str(OUT), str(REPO), str(REPO / 'research/vision/src'),
                str(REPO / 'research/eval/scripts')]

from research.multifloor.floors import floor_relation
from research.multifloor.sequence import supported_sequences
from research.multifloor.run import ground_metrics, json_safe
from backend import optimize, spatial_anchors


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(obj, key, default=float('inf')):
    value = obj.get(key)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else default


def visual_gate(row, labels):
    if floor_relation(row['query'], row['match'], labels) != 'same':
        return 'reject_floor'
    observation = row.get('sp', {})
    if (number(observation, 'inliers', 0) < 15 or
            sum(isinstance(value, (int, float)) and np.isfinite(value) and value >= 5
                for value in observation.get('per_cam', {}).values()) < 2):
        return 'reject_visual_support'
    if not (0 <= number(observation, 'diff_vs_odom_t_m') <= 5 and
            0 <= number(observation, 'diff_vs_odom_r_deg') <= 20):
        return 'hold_large_pose_disagreement'
    transform = np.asarray(observation.get('T', []), dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or
            not np.allclose(transform[3], [0, 0, 0, 1])):
        return 'reject_invalid_pnp'
    return 'pair_pass'


def main():
    started = time.monotonic()
    metadata = read_json(SUBMAPS / 'poses.json')
    times = np.asarray([row['t'] for row in metadata], dtype=float)
    poses = np.repeat(np.eye(4)[None], len(metadata), axis=0)
    poses[:, :3, 3] = [row['p'] for row in metadata]
    poses[:, :3, :3] = Rotation.from_quat([row['q'] for row in metadata]).as_matrix()
    floors = read_json(RELEASE / 'floor_hypotheses.json')
    measurements = read_json(RELEASE / 'ground_measurements.json')
    frozen_shortlist = read_json(BASE / 'comparison_20261016/visual_retrieval/shortlists.json')['four_joint']
    cache = {(row['query'], row['match']): row for row in read_json(RELEASE / 'candidates.json')}
    rows = [cache[(row['query'], row['match'])] for row in frozen_shortlist]
    assert len(rows) == 1100

    gates = Counter(visual_gate(row, floors['labels']) for row in rows)
    passing = []
    for row in rows:
        if visual_gate(row, floors['labels']) != 'pair_pass':
            continue
        clone = dict(row)
        clone['sp'] = dict(row['sp'], icp_T=row['sp']['T'])
        passing.append(clone)
    assert len(passing) == 42, gates
    supported = supported_sequences(passing, poses, times, floors['labels'], max_gap_sec=24.)
    anchors = spatial_anchors(supported, poses, times)
    assert len(supported) == 17 and len(anchors) == 16
    historical = read_json(BASE / 'comparison_20261016/no_icp_gate/summary.json')
    assert {(r['query'], r['match']) for r in anchors} == {
        (r['i'], r['j']) for r in historical['visual_anchor_edges']}
    print('INPUT', len(rows), 'PAIR_PASS', len(passing),
          'SEQUENCE_SUPPORTED', len(supported), 'ANCHORS', len(anchors), flush=True)

    corrected, summary = optimize(poses, times, supported, measurements, floors)
    before = ground_metrics(poses, measurements, floors['labels'])
    after = ground_metrics(corrected, measurements, floors['labels'])
    accepted = [edge for edge in summary['edges'] if edge['gnc_weight'] > .5]
    violations = []
    if not np.isfinite(corrected).all():
        violations.append('nonfinite_trajectory')
    for level in before:
        if after[level]['p90_range_m'] > before[level]['p90_range_m'] + .02:
            violations.append(f'floor_{level}_ground_range_worsened')
        if after[level]['mean_tilt_deg'] > before[level]['mean_tilt_deg'] + .5:
            violations.append(f'floor_{level}_tilt_worsened')
    for transition in floors['transitions']:
        delta = summary['floor_heights'][transition['to_level']] - summary['floor_heights'][transition['from_level']]
        if abs(delta - transition['delta_m']) > max(.8, 3 * transition['sigma_m']):
            violations.append('connection_height_changed')
        if abs(delta) < 1.8:
            violations.append('levels_collapsed')
    result = dict(
        method='four_view_no_symmetric_icp_gate_end_to_end',
        candidates=len(rows), visual_gate_counts=dict(gates),
        pair_pass=len(passing), sequence_supported=len(supported),
        spatial_anchors=len(anchors), accepted_loops=len(accepted),
        accepted_edges=accepted, all_optimized_edges=summary['edges'],
        ground_before=before, ground_after=after,
        floor_heights=summary['floor_heights'], quality_violations=violations,
        max_position_change_from_raw_m=float(np.max(np.linalg.norm(
            corrected[:, :3, 3] - poses[:, :3, 3], axis=1))),
        elapsed_s=time.monotonic() - started,
    )
    (OUT / 'summary.json').write_text(json.dumps(json_safe(result), indent=2))
    table = np.column_stack([times, corrected[:, :3, 3],
                             Rotation.from_matrix(corrected[:, :3, :3]).as_quat()])
    np.savetxt(OUT / 'optimized_poses.txt', table, fmt='%.9f')
    provenance = dict(inputs={str(path): sha256(path) for path in [
        SUBMAPS / 'poses.json', RELEASE / 'candidates.json',
        RELEASE / 'floor_hypotheses.json', RELEASE / 'ground_measurements.json',
        BASE / 'comparison_20261016/visual_retrieval/shortlists.json',
        BASE / 'comparison_20261016/no_icp_gate/summary.json',
        OUT / 'run.py', OUT / 'backend.py']})
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print('RESULT', json.dumps({key: result[key] for key in [
        'spatial_anchors', 'accepted_loops', 'quality_violations',
        'max_position_change_from_raw_m', 'elapsed_s']}), flush=True)


if __name__ == '__main__':
    main()
