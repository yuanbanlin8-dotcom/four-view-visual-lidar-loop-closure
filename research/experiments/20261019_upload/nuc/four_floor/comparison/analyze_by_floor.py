from pathlib import Path
from collections import Counter
import json,os,sys,numpy as np
from scipy.spatial.transform import Rotation

b=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
ex=b/'comparison_20261016';repo=Path('/home/nuc/five_floor_optimization_20261013')
sys.path[:0]=[str(repo),str(repo/'research/vision/src'),str(repo/'research/eval/scripts')]
from research.multifloor.verify import edge_gate
from research.multifloor.sequence import supported_sequences
sub=Path(str(b)+'_submaps');meta=json.loads((sub/'poses.json').read_text())
times=np.array([m['t'] for m in meta],float)
poses=np.repeat(np.eye(4)[None],len(meta),axis=0)
poses[:,:3,3]=[m['p'] for m in meta]
poses[:,:3,:3]=Rotation.from_quat([m['q'] for m in meta]).as_matrix()
floors=json.loads((b/'loop_latest_20261014/floor_hypotheses.json').read_text())
rank={x:i+1 for i,x in enumerate(np.argsort(floors['levels']))}
cache={(r['query'],r['match']):r for r in json.loads((b/'loop_latest_20261014/candidates.json').read_text())}
allrows=cache.copy()
for k in range(4):
 for r in json.loads((ex/'four_independent/workers'/str(k)/'candidates.json').read_text()):
  allrows[(r['query'],r['match'])]=r
short=json.loads((ex/'visual_retrieval/shortlists.json').read_text())
for mode in ('four_joint','four_independent'):
 rows=[allrows[(r['query'],r['match'])] for r in short[mode]]
 passing=[r for r in rows if edge_gate(r,floors['labels'])=='pair_pass']
 supported=supported_sequences(passing,poses,times,floors['labels'],max_gap_sec=24.)
 print('MODE',mode)
 for name,group in [('passed',passing),('supported',supported)]:
  print(name,dict(Counter(rank[floors['labels'][r['query']]] for r in group)))
  for r in group:
   if rank[floors['labels'][r['query']]]==4:
    print('FLOOR4',name,r['query'],r['match'],r['sp']['inliers'],r.get('sequence_id'))
