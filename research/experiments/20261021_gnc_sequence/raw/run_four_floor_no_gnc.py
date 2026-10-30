"""Isolated four-floor GNC ablation; never edits the frozen pipeline."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
from scipy.spatial.transform import Rotation

base = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
release = base / 'loop_latest_20261014'
sub = base.parent / 'four_floor_20261013_160929_submaps'
repo = Path('/home/nuc/five_floor_optimization_20261013')
out = base / 'gnc_ablation_20261021'
out.mkdir(exist_ok=True)
os.environ.update(
    SEEKER_RESEARCH_ROOT=str(repo / 'research'),
    SEEKER_SUBMAP_ROOT=str(sub),
    SEEKER_RUNTIME_CONFIG=str(base / 'loop_inputs/runtime.yaml'),
    SEEKER_CAMERA_CONFIG=str(repo / 'mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'),
    SEEKER_VISION_ROOT=str(release / 'vision'),
    SEEKER_MASK='v2',
    OPENBLAS_NUM_THREADS='2',
    OMP_NUM_THREADS='2',
)
sys.path[:0] = [str(repo), str(repo / 'research/vision/src'), str(repo / 'research/eval/scripts')]

from research.multifloor.verify import edge_gate
from research.multifloor.sequence import supported_sequences
from research.multifloor.backend import optimize as original_optimize
from research.multifloor.run import ground_metrics, json_safe

metadata = json.loads((sub / 'poses.json').read_text())
times = np.asarray([m['t'] for m in metadata], float)
poses = np.repeat(np.eye(4)[None], len(metadata), axis=0)
poses[:, :3, 3] = [m['p'] for m in metadata]
poses[:, :3, :3] = Rotation.from_quat([m['q'] for m in metadata]).as_matrix()
floors = json.loads((release / 'floor_hypotheses.json').read_text())
measurements = json.loads((release / 'ground_measurements.json').read_text())
records = json.loads((release / 'candidates.json').read_text())
passing = [r for r in records if edge_gate(r, floors['labels']) == 'pair_pass']
supported = supported_sequences(passing, poses, times, floors['labels'], max_gap_sec=24.)
assert len(poses) == 243 and len(passing) == 29 and len(supported) == 28

# Load a copy of the frozen function and skip only its GNC reweighting loop.
backend_path = repo / 'research/multifloor/backend.py'
source = backend_path.read_text()
old = '    if loops:\n        cutoff=16.81;res=residuals(estimate)'
new = '    if False:  # GNC ablation\n        cutoff=16.81;res=residuals(estimate)'
assert source.count(old) == 1
module = types.ModuleType('backend_no_gnc')
module.__file__ = str(backend_path)
exec(compile(source.replace(old, new), str(backend_path), 'exec'), module.__dict__)

def run(name, optimizer, rows):
    start = time.monotonic()
    corrected, audit = optimizer(poses, times, rows, measurements, floors)
    accepted = [e for e in audit['edges'] if e['gnc_weight'] > .5]
    result = dict(
        mode=name, dataset='20261013_160929', nodes=len(poses),
        pair_pass=len(passing), passed_to_backend=len(rows),
        spatial_anchor_count=audit['spatial_anchor_count'], pcm=audit['pcm'],
        accepted_loops=audit['loop_count'], accepted_edges=accepted,
        ground_after=ground_metrics(corrected, measurements, floors['labels']),
        floor_heights=audit['floor_heights'], elapsed_seconds=time.monotonic()-start,
    )
    (out / f'{name}_summary.json').write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
    np.savetxt(out / f'{name}_poses.txt',
               np.column_stack([times, corrected[:, :3, 3], Rotation.from_matrix(corrected[:, :3, :3]).as_quat()]),
               fmt='%.9f')
    print('RESULT', name, json.dumps({k: result[k] for k in
          ('pair_pass', 'passed_to_backend', 'spatial_anchor_count', 'pcm', 'accepted_loops', 'elapsed_seconds')}), flush=True)
    return corrected, result

baseline_poses, baseline = run('baseline_24s', original_optimize, supported)
frozen = json.loads((base / 'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())
frozen_poses = np.loadtxt(base / 'sequence_gap_ablation_20261014/gap_24_poses.txt')
assert baseline['accepted_loops'] == frozen['accepted_loops'] == 26
assert {(e['i'], e['j']) for e in baseline['accepted_edges']} == {
    (e['i'], e['j']) for e in frozen['accepted_edges']}
assert np.max(np.linalg.norm(baseline_poses[:, :3, 3]-frozen_poses[:, 1:4], axis=1)) < 1e-3

no_gnc_poses, no_gnc = run('no_gnc_24s', module.optimize, supported)

# Remove sequence support only as an additional stress check, preserving the
# same candidate and edge metadata used by the existing no-sequence ablation.
by_pair = {(r['query'], r['match']): r for r in supported}
no_sequence = [by_pair[(r['query'], r['match'])] if (r['query'], r['match']) in by_pair else
               dict(r, sequence_id=-1-k, sequence_pairs=1,
                    sequence_direction=None, sequence_span_m=None)
               for k, r in enumerate(passing)]
no_seq_poses, no_seq = run('no_sequence_with_gnc', original_optimize, no_sequence)
no_seq_no_gnc_poses, no_seq_no_gnc = run('no_sequence_no_gnc', module.optimize, no_sequence)

def compare(a, b, a_poses, b_poses):
    ea = {(e['i'], e['j']) for e in a['accepted_edges']}
    eb = {(e['i'], e['j']) for e in b['accepted_edges']}
    delta = b_poses[:, :3, 3] - a_poses[:, :3, 3]
    return dict(added_edges=sorted(map(list, eb-ea)), removed_edges=sorted(map(list, ea-eb)),
                max_position_change_m=float(np.max(np.linalg.norm(delta, axis=1))),
                median_position_change_m=float(np.median(np.linalg.norm(delta, axis=1))),
                max_height_change_m=float(np.max(np.abs(delta[:, 2]))))

comparison = dict(
    with_sequence=compare(baseline, no_gnc, baseline_poses, no_gnc_poses),
    without_sequence=compare(no_seq, no_seq_no_gnc, no_seq_poses, no_seq_no_gnc_poses),
)
(out / 'comparison.json').write_text(json.dumps(comparison, indent=2))
print('COMPARISON', json.dumps(comparison), flush=True)
