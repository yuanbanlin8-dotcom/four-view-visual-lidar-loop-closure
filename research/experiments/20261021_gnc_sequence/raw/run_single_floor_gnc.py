"""Replay current sequence rule and factorial PCM/GNC ablation on floor-1 candidates."""
from pathlib import Path
import csv
import json
import os
import sys
import time
import types

import numpy as np
from scipy.spatial.transform import Rotation

repo = Path('/home/nuc/five_floor_optimization_20261013')
research = repo / 'research'
out = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929/gnc_ablation_20261021/single_floor')
out.mkdir(parents=True, exist_ok=True)
os.environ.update(SEEKER_RESEARCH_ROOT=str(research), OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
sys.path[:0] = [str(repo), str(research / 'vision/src')]

from research.multifloor.sequence import supported_sequences

archive = json.loads((research / 'vision/results/vpr2_candidates_v2_z.json').read_text())
data = np.load(research / 'backend/results/poses.npz')
poses, times = data['T0'], data['t']
assert poses.shape == (134, 4, 4)

def pair_pass(r):
    sp = r['sp']
    return (sp.get('inliers', 0) >= 10 and sp.get('icp_fitness', 0) >= .25
            and sp.get('icp_corr_t', 9) <= .5 and sp.get('icp_corr_r', 99) <= 5)

passing = [r for r in archive if pair_pass(r)]
assert len(passing) == 55
supported = supported_sequences(passing, poses, times, [0]*len(poses), max_gap_sec=24.)
assert len(supported) == 28
by_pair = {(r['query'], r['match']): r for r in supported}
no_sequence = [by_pair[(r['query'],r['match'])] if (r['query'],r['match']) in by_pair else
               dict(r, sequence_id=-1-k, sequence_pairs=1,
                    sequence_direction=None, sequence_span_m=None)
               for k,r in enumerate(passing)]

# Use the original 55-edge ICP covariance/degeneracy audit. Recomputing it
# would introduce a different geometry fit into this backend-only comparison.
cached = {}
with (research / 'vision/results/vision_loops_pcm.csv').open() as fh:
    for row in csv.DictReader(fh):
        if row['in_input55'] != '1':
            continue
        pair = (int(row['query_node']), int(row['match_node']))
        weak_map = np.fromstring(row['weak_dir_map'], sep=' ')
        weak_j = poses[pair[1],:3,:3].T @ weak_map
        cov = np.fromstring(row['cov_6x6_rot_trans'], sep=' ').reshape(6,6)
        cached[pair] = dict(cov_6x6=cov, degenerate=row['degenerate']=='1',
                            weak_dir_j=weak_j)
assert len(cached) == 55
fake = types.ModuleType('degeneracy')
fake.analyse = lambda i,j,*args,**kwargs: cached[(i,j)]
sys.modules['degeneracy'] = fake

backend_path = research / 'multifloor/backend.py'
source = backend_path.read_text()
gnc_line = '    if loops:\n        cutoff=16.81;res=residuals(estimate)'
pcm_line = '        clique,_,_=pcm.consistent_set([edges[k] for k in indices],p,length)'
assert source.count(gnc_line) == 1 and source.count(pcm_line) == 1

def optimizer(use_pcm, use_gnc):
    edited = source
    if not use_pcm:
        edited = edited.replace(pcm_line, '        clique=list(range(len(indices)))  # PCM ablation')
    if not use_gnc:
        edited = edited.replace(gnc_line, '    if False:  # GNC ablation\n        cutoff=16.81;res=residuals(estimate)')
    mod = types.ModuleType('backend_variant')
    mod.__file__ = str(backend_path)
    exec(compile(edited, str(backend_path), 'exec'), mod.__dict__)
    return mod.optimize

floor = dict(labels=[0]*len(poses), levels=[], transitions=[])
results = {}
for name, rows, use_pcm, use_gnc in [
    ('seq_pcm_gnc', supported, True, True),
    ('seq_pcm_no_gnc', supported, True, False),
    ('seq_no_pcm_gnc', supported, False, True),
    ('seq_no_pcm_no_gnc', supported, False, False),
    ('all_pcm_gnc', no_sequence, True, True),
    ('all_pcm_no_gnc', no_sequence, True, False),
    ('all_no_pcm_gnc', no_sequence, False, True),
    ('all_no_pcm_no_gnc', no_sequence, False, False),
]:
    start = time.monotonic()
    try:
        corrected, audit = optimizer(use_pcm, use_gnc)(poses, times, rows, [], floor, use_ground=False)
        accepted = sorted((e['i'],e['j']) for e in audit['edges'] if e['gnc_weight'] > .5)
        result = dict(name=name, input_candidates=len(rows), spatial_anchors=audit['spatial_anchor_count'],
                      pcm=audit['pcm'], final_loops=audit['loop_count'], accepted_edges=accepted,
                      finite=bool(np.isfinite(corrected).all()),
                      max_deviation_from_raw_m=float(np.max(np.linalg.norm(corrected[:,:3,3]-poses[:,:3,3],axis=1))),
                      elapsed_seconds=time.monotonic()-start)
        results[name] = result
        np.save(out / f'{name}_poses.npy', corrected)
        print('RESULT', name, json.dumps({k:result[k] for k in ('input_candidates','spatial_anchors','pcm','final_loops','finite','max_deviation_from_raw_m','elapsed_seconds')}),flush=True)
    except Exception as exc:
        results[name] = dict(error=repr(exc))
        print('ERROR', name, repr(exc), flush=True)
    (out/'results.json').write_text(json.dumps(results, indent=2))

for a,b in [('seq_pcm_gnc','seq_pcm_no_gnc'),('all_pcm_gnc','all_pcm_no_gnc'),
            ('seq_pcm_gnc','seq_no_pcm_gnc'),('all_pcm_gnc','all_no_pcm_gnc')]:
    if 'error' in results[a] or 'error' in results[b]: continue
    pa,pb = np.load(out/f'{a}_poses.npy'), np.load(out/f'{b}_poses.npy')
    delta = pb[:,:3,3]-pa[:,:3,3]
    print('COMPARISON',a,b,'max_m',float(np.max(np.linalg.norm(delta,axis=1))),
          'median_m',float(np.median(np.linalg.norm(delta,axis=1))), flush=True)
