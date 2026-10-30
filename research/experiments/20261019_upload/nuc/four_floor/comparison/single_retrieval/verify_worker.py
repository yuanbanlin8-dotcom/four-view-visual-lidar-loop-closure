from pathlib import Path
import os,sys,json,argparse,time
parser=argparse.ArgumentParser();parser.add_argument('worker',type=int);args=parser.parse_args()
base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
work=base/'comparison_20261016/single_retrieval/workers'/str(args.worker)
repo=Path('/home/nuc/five_floor_optimization_20261013')
os.environ.update(SEEKER_RESEARCH_ROOT=str(repo/'research'),SEEKER_SUBMAP_ROOT=str(base.parent/'four_floor_20261013_160929_submaps'),SEEKER_RUNTIME_CONFIG=str(base/'loop_inputs/runtime.yaml'),SEEKER_CAMERA_CONFIG=str(repo/'mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'),SEEKER_VISION_ROOT=str(work/'vision'),SEEKER_MASK='v2',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TORCH_HOME='/home/nuc/.cache/torch')
sys.path[:0]=[str(repo),str(repo/'research/vision/src'),str(repo/'research/eval/scripts')]
import numpy as np,torch,cv2
torch.set_num_interop_threads(1);cv2.setNumThreads(1)
from research.multifloor.verify import Verifier
from research.multifloor.run import save
with np.load(base/'vision_input/global_dinov2.npz') as z:desc=z['desc']
verifier=Verifier(desc.shape,base/'loop_latest_20261014/vision/data/featcache/v2_horizontal3')
torch.set_num_threads(2)
rows=json.loads((work/'tasks.json').read_text());records=[];start=time.monotonic()
for row in rows:
 i,j=row['query'],row['match'];similarity=np.einsum('avd,bwd->avbw',desc[i],desc[j])
 shift=int(row['shift'])
 record=dict(row,shift=shift,sp=verifier.verify(i,j,shift,similarity));records.append(record)
 if len(verifier.mr.cache)>600:verifier.mr.cache.clear()
 if len(verifier.pn._sub)>80:verifier.pn._sub.clear()
 save(work/'candidates.json',records)
 print('verified',args.worker,i,j,record['sp'].get('inliers',0),len(records),'/',len(rows),flush=True)
save(work/'complete.json',dict(worker=args.worker,records=len(records),elapsed_seconds=time.monotonic()-start))
