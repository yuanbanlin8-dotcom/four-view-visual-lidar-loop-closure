"""Fixed single-camera retrieval scores with a shared four-camera verifier/backend."""
from pathlib import Path
from collections import Counter
import json,os,sys,time
import numpy as np
from scipy.spatial.transform import Rotation

base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
root=base/'comparison_20261016';work=root/'single_retrieval'
release=base/'loop_latest_20261014';repo=Path('/home/nuc/five_floor_optimization_20261013')
sub=Path(str(base)+'_submaps')
os.environ.update(SEEKER_RESEARCH_ROOT=str(repo/'research'),SEEKER_SUBMAP_ROOT=str(sub),
                  SEEKER_RUNTIME_CONFIG=str(base/'loop_inputs/runtime.yaml'),
                  SEEKER_CAMERA_CONFIG=str(repo/'mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'),
                  SEEKER_VISION_ROOT=str(release/'vision'),SEEKER_MASK='v2',
                  OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
sys.path[:0]=[str(repo),str(repo/'research/vision/src'),str(repo/'research/eval/scripts')]
from research.multifloor.verify import edge_gate
from research.multifloor.sequence import supported_sequences
from research.multifloor.backend import optimize
from research.multifloor.run import ground_metrics,json_safe

assert all((work/'workers'/str(k)/'complete.json').exists() for k in range(8))
allrows={(r['query'],r['match']):r for r in json.loads((release/'candidates.json').read_text())}
for folder,count in ((root/'four_independent/workers',4),(work/'workers',8)):
 for k in range(count):
  for r in json.loads((folder/str(k)/'candidates.json').read_text()):
   key=(r['query'],r['match'])
   assert key not in allrows,key
   allrows[key]=r
assert len(allrows)==1919,len(allrows)
short=json.loads((root/'visual_retrieval/shortlists.json').read_text())
metadata=json.loads((sub/'poses.json').read_text());times=np.array([m['t'] for m in metadata],float)
poses=np.repeat(np.eye(4)[None],len(metadata),axis=0)
poses[:,:3,3]=[m['p'] for m in metadata]
poses[:,:3,:3]=Rotation.from_quat([m['q'] for m in metadata]).as_matrix()
floors=json.loads((release/'floor_hypotheses.json').read_text())
measurements=json.loads((release/'ground_measurements.json').read_text())
rank={label:index+1 for index,label in enumerate(np.argsort(floors['levels']).tolist())}
before=ground_metrics(poses,measurements,floors['labels'])
full=json.loads((base/'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())
full_accepted={(e['i'],e['j']) for e in full['accepted_edges'] if e['gnc_weight']>.5}
for mode in ('single_c0','single_c1','single_c2','single_c3'):
 start=time.monotonic()
 rows=[allrows[(r['query'],r['match'])] for r in short[mode]]
 gates=Counter(edge_gate(r,floors['labels']) for r in rows)
 passing=[r for r in rows if edge_gate(r,floors['labels'])=='pair_pass']
 supported=supported_sequences(passing,poses,times,floors['labels'],max_gap_sec=24.)
 corrected,summary=optimize(poses,times,supported,measurements,floors)
 after=ground_metrics(corrected,measurements,floors['labels'])
 violations=[]
 for level in before:
  if after[level]['p90_range_m']>before[level]['p90_range_m']+.02:violations.append('floor_'+level+'_ground_range_worsened')
  if after[level]['mean_tilt_deg']>before[level]['mean_tilt_deg']+.5:violations.append('floor_'+level+'_tilt_worsened')
 for tr in floors['transitions']:
  delta=summary['floor_heights'][tr['to_level']]-summary['floor_heights'][tr['from_level']]
  if abs(delta-tr['delta_m'])>max(.8,3*tr['sigma_m']):violations.append('connection_height_changed')
  if abs(delta)<1.8:violations.append('levels_collapsed')
 if not np.isfinite(corrected).all():violations.append('nonfinite_trajectory')
 accepted=[e for e in summary['edges'] if e['gnc_weight']>.5]
 actual={(e['i'],e['j']) for e in accepted}
 out=dict(method=mode+'_retrieval_same_multicamera_verifier',candidates=len(rows),
          gate_counts=dict(gates),pair_pass=len(passing),sequence_supported=len(supported),
          accepted_loops=summary['loop_count'],
          accepted_by_floor={str(f):sum(rank[e['floor']]==f for e in accepted) for f in range(1,5)},
          accepted_edges=accepted,accepted_overlap_full=len(actual&full_accepted),
          accepted_unique=len(actual-full_accepted),full_unique=len(full_accepted-actual),
          pcm=summary['pcm'],ground_before=before,ground_after=after,
          quality_violations=violations,floor_heights=summary['floor_heights'],
          elapsed_seconds=time.monotonic()-start)
 (work/(mode+'_summary.json')).write_text(json.dumps(json_safe(out),indent=2))
 np.savetxt(work/(mode+'_optimized_poses.txt'),np.column_stack([times,corrected[:,:3,3],
            Rotation.from_matrix(corrected[:,:3,:3]).as_quat()]),fmt='%.9f')
 print('FINAL',mode,json.dumps({k:out[k] for k in ['pair_pass','sequence_supported',
       'accepted_loops','accepted_by_floor','accepted_overlap_full','accepted_unique',
       'quality_violations','elapsed_seconds']}),flush=True)
