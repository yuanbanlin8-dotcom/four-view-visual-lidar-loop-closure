"""Detection-only ablation: keep common retrieval/visual PnP/odom/floor gates, omit symmetric LiDAR ICP gate."""
from pathlib import Path
from collections import Counter
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation

base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
release=base/'loop_latest_20261014'
out=base/'comparison_20261016/no_icp_gate'
out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'/home/nuc/five_floor_optimization_20261013')
from research.multifloor.verify import edge_gate
from research.multifloor.sequence import supported_sequences
from research.multifloor.backend import spatial_anchors

short=json.loads((base/'comparison_20261016/visual_retrieval/shortlists.json').read_text())['four_joint']
cache={(r['query'],r['match']):r for r in json.loads((release/'candidates.json').read_text())}
rows=[cache[(r['query'],r['match'])] for r in short]
floors=json.loads((release/'floor_hypotheses.json').read_text())['labels']
raw=np.loadtxt(release/'raw_poses.txt')
times=raw[:,0]
poses=np.repeat(np.eye(4)[None],len(raw),axis=0)
poses[:,:3,3]=raw[:,1:4]
poses[:,:3,:3]=Rotation.from_quat(raw[:,4:8]).as_matrix()

def number(o,key,default=float('inf')):
    v=o.get(key)
    return float(v) if isinstance(v,(int,float)) and np.isfinite(v) else default

def visual_gate(r):
    from research.multifloor.floors import floor_relation
    i,j=r['query'],r['match']
    if floor_relation(i,j,floors)!='same':return 'reject_floor'
    o=r.get('sp',{})
    if number(o,'inliers',0)<15 or sum(isinstance(v,(int,float)) and np.isfinite(v) and v>=5 for v in o.get('per_cam',{}).values())<2:
        return 'reject_visual_support'
    if not (0<=number(o,'diff_vs_odom_t_m')<=5 and 0<=number(o,'diff_vs_odom_r_deg')<=20):
        return 'hold_large_pose_disagreement'
    T=np.asarray(o.get('T',[]),float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        return 'reject_invalid_pnp'
    return 'pair_pass'

visual=[r for r in rows if visual_gate(r)=='pair_pass']
full=[r for r in rows if edge_gate(r,floors)=='pair_pass']
# Reuse sequence code with the visual PnP transform, without referencing ICP.
visual_pnp=[]
for r in visual:
    clone=dict(r)
    clone['sp']=dict(r['sp'],icp_T=r['sp']['T'])
    visual_pnp.append(clone)
supported=supported_sequences(visual_pnp,poses,times,floors,max_gap_sec=24.)
anchors=spatial_anchors(supported,poses,times)
def keys(rs):return {(r['query'],r['match']) for r in rs}
full_summary=json.loads((base/'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())
accepted={(e['i'],e['j']) for e in full_summary['accepted_edges'] if e['gnc_weight']>.5}
def info(r):
    s=r['sp'];i,j=r['query'],r['match']
    return {'i':i,'j':j,'visual_inliers':s.get('inliers'),'per_cam':s.get('per_cam'),
            'pnp_odom_diff_m':s.get('diff_vs_odom_t_m'),'pnp_odom_diff_deg':s.get('diff_vs_odom_r_deg'),
            'icp_fitness':s.get('icp_fitness'),'icp_rmse':s.get('icp_rmse'),
            'reverse_fitness':s.get('reverse_fitness'),'reverse_rmse':s.get('reverse_rmse'),
            'full_gate':edge_gate(r,floors),'full_accepted':(i,j) in accepted,
            'sequence_pairs':r.get('sequence_pairs')}
report={'method':'same joint DINO+multicamera+LiDAR-depth PnP; symmetric LiDAR ICP gate removed',
        'candidate_count':len(rows),'visual_gate_counts':dict(Counter(visual_gate(r) for r in rows)),
        'full_gate_counts':dict(Counter(edge_gate(r,floors) for r in rows)),
        'visual_pair_pass':len(visual),'full_pair_pass':len(full),
        'visual_sequence_supported':len(supported),'visual_spatial_anchors':len(anchors),
        'full_final_accepted':len(accepted),
        'visual_only_pair_pass':[info(r) for r in visual if (r['query'],r['match']) not in keys(full)],
        'visual_sequence_edges':[info(r) for r in supported],
        'visual_anchor_edges':[info(r) for r in anchors]}
(out/'summary.json').write_text(json.dumps(report,indent=2))
print('SUMMARY',{k:report[k] for k in ['candidate_count','visual_gate_counts','full_gate_counts','visual_pair_pass','full_pair_pass','visual_sequence_supported','visual_spatial_anchors','full_final_accepted']})
print('VISUAL_ONLY_PAIR_KEYS',[(r['i'],r['j'],r['full_gate']) for r in report['visual_only_pair_pass']])
print('VISUAL_ANCHOR_KEYS',[(r['i'],r['j'],r['full_gate']) for r in report['visual_anchor_edges']])
