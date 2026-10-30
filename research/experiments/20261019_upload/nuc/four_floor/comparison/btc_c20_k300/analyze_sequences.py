from pathlib import Path
import csv,json,os,sys,numpy as np
from scipy.spatial.transform import Rotation

b=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
sub=Path(str(b)+'_submaps');repo=Path('/home/nuc/five_floor_optimization_20261013')
sys.path[:0]=[str(repo),str(repo/'research/vision/src'),str(repo/'research/eval/scripts')]
from research.multifloor.sequence import supported_sequences
metadata=json.loads((sub/'poses.json').read_text());times=np.array([m['t'] for m in metadata],float)
poses=np.repeat(np.eye(4)[None],len(metadata),axis=0)
poses[:,:3,3]=[m['p'] for m in metadata]
rr=Rotation.from_quat([m['q'] for m in metadata])
poses[:,:3,:3]=rr.as_matrix() if hasattr(rr,'as_matrix') else rr.as_dcm()
labels=json.loads((b/'loop_latest_20261014/floor_hypotheses.json').read_text())['labels']
for candidate_num,useful_corner_num in ((20,300),(20,500),(50,300),(50,500)):
 raw=list(csv.DictReader((b/f'comparison_20261016/btc_c{candidate_num}_k{useful_corner_num}/candidates.csv').open()))
 for inverse in (False,True):
  for limit in (None,.5,.3,.18):
   rows=[]
   for r in raw:
    i,j=int(r['i']),int(r['j'])
    if labels[i] is None or labels[i]!=labels[j] or (limit is not None and float(r['rms'])>limit):continue
    T=np.eye(4);T[:3,3]=[float(r[k]) for k in 'tx ty tz'.split()]
    rr=Rotation.from_quat([float(r[k]) for k in 'qx qy qz qw'.split()])
    T[:3,:3]=rr.as_matrix() if hasattr(rr,'as_matrix') else rr.as_dcm()
    if inverse:T=np.linalg.inv(T)
    rows.append({'query':i,'match':j,'sp':{'icp_T':T.tolist(),'inliers':int(r['planes'])}})
   supported=supported_sequences(rows,poses,times,labels,max_gap_sec=24)
   print('BTC_SEQ',candidate_num,useful_corner_num,'inverse' if inverse else 'direct',limit,
         'all_candidates',len(raw),'same_floor_candidates',len(rows),
         'supported_pairs',len(supported),'supported_unique',len({(r['query'],r['match']) for r in supported}))
   if supported:
    for r in supported:
     source=next(x for x in raw if int(x['i'])==r['query'] and int(x['j'])==r['match'])
     T=np.array(r['sp']['icp_T']);expected=np.linalg.inv(poses[r['match']])@poses[r['query']]
     d=np.linalg.inv(expected)@T
     print('BTC_SUPPORTED',r['query'],r['match'],'floor',labels[r['query']],
           'score',source['score'],'rms',source['rms'],'planes',source['planes'],
           'odom_diff_m',float(np.linalg.norm(d[:3,3])),
           'odom_diff_deg',float(np.degrees(Rotation.from_matrix(d[:3,:3]).magnitude())))
