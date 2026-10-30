"""Independent four-floor backend ablation; keeps frozen retrieval and verifier data."""
from pathlib import Path
import json,os,sys,time
import numpy as np
from scipy.spatial.transform import Rotation

base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
release=base/'loop_latest_20261014'
sub=base.parent/'four_floor_20261013_160929_submaps'
out=base/'backend_ablation_pcm_ground_20261017'
out.mkdir(exist_ok=True)
repo=Path('/home/nuc/five_floor_optimization_20261013')
os.environ.update(SEEKER_RESEARCH_ROOT=str(repo/'research'),SEEKER_SUBMAP_ROOT=str(sub),
                  SEEKER_RUNTIME_CONFIG=str(base/'loop_inputs/runtime.yaml'),
                  SEEKER_CAMERA_CONFIG=str(repo/'mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'),
                  SEEKER_VISION_ROOT=str(release/'vision'),SEEKER_MASK='v2',
                  OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
sys.path[:0]=[str(repo),str(repo/'research/vision/src'),str(repo/'research/eval/scripts')]
from research.multifloor.verify import edge_gate
from research.multifloor.sequence import supported_sequences
from research.multifloor.backend import optimize
from research.multifloor.run import ground_metrics,json_safe
import pcm

metadata=json.loads((sub/'poses.json').read_text())
times=np.asarray([m['t'] for m in metadata],float)
poses=np.repeat(np.eye(4)[None],len(metadata),axis=0)
poses[:,:3,3]=[m['p'] for m in metadata]
poses[:,:3,:3]=Rotation.from_quat([m['q'] for m in metadata]).as_matrix()
floors=json.loads((release/'floor_hypotheses.json').read_text())
measurements=json.loads((release/'ground_measurements.json').read_text())
records=json.loads((release/'candidates.json').read_text())
passing=[r for r in records if edge_gate(r,floors['labels'])=='pair_pass']
supported=supported_sequences(passing,poses,times,floors['labels'],max_gap_sec=24.)
assert len(poses)==243 and len(passing)==29 and len(supported)==28,(len(poses),len(passing),len(supported))
reference=json.loads((base/'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())
assert reference['accepted_loops']==26
fullposes=np.loadtxt(base/'sequence_gap_ablation_20261014/gap_24_poses.txt')
rank={label:index+1 for index,label in enumerate(np.argsort(floors['levels']).tolist())}

def summarize(mode,corrected,summary,elapsed):
 accepted=[e for e in summary['edges'] if e['gnc_weight']>.5]
 metrics=ground_metrics(corrected,measurements,floors['labels'])
 before=ground_metrics(poses,measurements,floors['labels'])
 levels=sorted(metrics,key=lambda k:rank[int(k)])
 centers=[metrics[k]['center_m'] for k in levels]
 shift=corrected[:,:3,3]-poses[:,:3,3]
 difffull=corrected[:,:3,3]-fullposes[:,1:4]
 result=dict(mode=mode,dataset='20261013_160929',nodes=len(poses),candidate_count=len(records),
             pair_pass=len(passing),sequence_supported=len(supported),
             spatial_anchor_count=summary['spatial_anchor_count'],pcm=summary['pcm'],
             accepted_loops=summary['loop_count'],
             accepted_by_relative_floor={str(i):sum(rank[e['floor']]==i for e in accepted) for i in range(1,5)},
             accepted_edges=accepted,ground_enabled=summary['ground_enabled'],
             floor_heights=summary['floor_heights'],ground_before=before,ground_after=metrics,
             observed_floor_centers_m=centers,
             observed_adjacent_floor_gaps_m=np.diff(centers).tolist(),
             shift_from_input_m=dict(median=float(np.median(np.linalg.norm(shift,axis=1))),
                                     p95=float(np.percentile(np.linalg.norm(shift,axis=1),95)),
                                     maximum=float(np.max(np.linalg.norm(shift,axis=1))),
                                     maximum_vertical=float(np.max(np.abs(shift[:,2])))),
             difference_vs_full_m=dict(median=float(np.median(np.linalg.norm(difffull,axis=1))),
                                       p95=float(np.percentile(np.linalg.norm(difffull,axis=1),95)),
                                       maximum=float(np.max(np.linalg.norm(difffull,axis=1))),
                                       maximum_vertical=float(np.max(np.abs(difffull[:,2])))),
             elapsed_seconds=elapsed)
 (out/f'{mode}_summary.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False))
 table=np.column_stack([times,corrected[:,:3,3],Rotation.from_matrix(corrected[:,:3,:3]).as_quat()])
 np.savetxt(out/f'{mode}_poses.txt',table,fmt='%.9f')
 print('RESULT',mode,json.dumps({k:result[k] for k in ('accepted_loops','accepted_by_relative_floor','pcm','observed_floor_centers_m','observed_adjacent_floor_gaps_m','shift_from_input_m','difference_vs_full_m','elapsed_seconds')}),flush=True)
 return result

start=time.monotonic()
baseline_poses,baseline=optimize(poses,times,supported,measurements,floors)
assert baseline['loop_count']==reference['accepted_loops']
assert {(e['i'],e['j']) for e in baseline['edges'] if e['gnc_weight']>.5}=={(e['i'],e['j']) for e in reference['accepted_edges']}
assert np.max(np.linalg.norm(baseline_poses[:,:3,3]-fullposes[:,1:4],axis=1))<1e-3
summarize('full_reproduction',baseline_poses,baseline,time.monotonic()-start)

original_pcm=pcm.consistent_set
def bypass_pcm(edges,p,length,*args,**kwargs):
 return list(range(len(edges))),None,None
pcm.consistent_set=bypass_pcm
try:
 start=time.monotonic()
 no_pcm_poses,no_pcm=optimize(poses,times,supported,measurements,floors)
 summarize('no_pcm',no_pcm_poses,no_pcm,time.monotonic()-start)
finally:
 pcm.consistent_set=original_pcm

start=time.monotonic()
no_ground_poses,no_ground=optimize(poses,times,supported,measurements,floors,use_ground=False)
summarize('no_ground_factors',no_ground_poses,no_ground,time.monotonic()-start)
print('DONE',flush=True)
