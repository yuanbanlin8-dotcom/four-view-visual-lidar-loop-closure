from pathlib import Path
import json
import sys

base = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
comp = base / 'comparison_20261016'
short = json.loads((comp / 'visual_retrieval/shortlists.json').read_text())
release = base / 'loop_latest_20261014'
sys.path.insert(0, '/home/nuc/five_floor_optimization_20261013')
from research.multifloor.verify import edge_gate
from research.multifloor.sequence import supported_sequences
from research.multifloor.backend import spatial_anchors
import numpy as np
from scipy.spatial.transform import Rotation

floors = json.loads((release / 'floor_hypotheses.json').read_text())['labels']
records = {(r['query'], r['match']): r for r in json.loads((release / 'candidates.json').read_text())}
for folder, count in ((comp/'four_independent/workers', 4), (comp/'single_retrieval/workers', 8)):
    for k in range(count):
        for r in json.loads((folder/str(k)/'candidates.json').read_text()):
            records[(r['query'], r['match'])] = r

raw = np.loadtxt(release/'raw_poses.txt')
times = raw[:,0]
poses = np.repeat(np.eye(4)[None],len(raw),axis=0)
poses[:,:3,3] = raw[:,1:4]
poses[:,:3,:3] = Rotation.from_quat(raw[:,4:8]).as_matrix()
joint_rows = [records[(r['query'],r['match'])] for r in short['four_joint']]
passing = [r for r in joint_rows if edge_gate(r,floors)=='pair_pass']
supported = supported_sequences(passing,poses,times,floors,max_gap_sec=24.)
anchors = spatial_anchors(supported,poses,times)
print('JOINT_STAGES',len(passing),len(supported),len(anchors))
def near_117(rows):
    return [(r['query'],r['match'],r['sp'].get('inliers')) for r in rows if r['query']==117]
print('Q117_PASS',near_117(passing))
print('Q117_SEQ',near_117(supported))
print('Q117_ANCHOR',near_117(anchors))
for r in passing:
    if 109 <= r['query'] <= 120:
        key=(r['query'],r['match'])
        print('NEAR117',key,'dt_query',round(float(times[r['query']]-times[117]),1),
              'dt_match',round(float(times[r['match']]-times[97]),1),
              'inliers',r['sp']['inliers'],'seq',any((s['query'],s['match'])==key for s in supported))

print('MODES', {k:len(v) for k,v in short.items()})
for key in [(69,45),(74,41),(78,37),(117,96),(117,97)]:
    mode_info = {}
    for mode, rows in short.items():
        hit = next((r for r in rows if (r['query'],r['match']) == key), None)
        mode_info[mode] = None if hit is None else {'rank':hit.get('rank'),'score':hit.get('score')}
    rec = records.get(key)
    print('EDGE', key, 'shortlists', mode_info,
          'gate', edge_gate(rec,floors) if rec else 'not_verified',
          'joint_sequence', any((r['query'],r['match'])==key for r in supported),
          'joint_anchor', any((r['query'],r['match'])==key for r in anchors))
