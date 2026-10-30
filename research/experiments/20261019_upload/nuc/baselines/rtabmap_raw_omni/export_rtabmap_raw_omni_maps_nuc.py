from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/home/nuc/omnilivo_floor_loop_trial/external_baselines_20261017/four_floor_rtabmap_raw_omni')
sub=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929_submaps')
out=root/'preserved_runs_20261017'
f=json.loads(Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929/loop_latest_20261014/floor_hypotheses.json').read_text())
labels=np.array([-1 if x is None else int(x) for x in f['labels']]);rank={int(v):int(i+1) for i,v in enumerate(np.argsort(f['levels']))}
count=sum(np.load(sub/f'{i:04d}.npy',mmap_mode='r').shape[0] for i in range(243))
result={}
for mode in (0,1,2,3,4):
 stem='rtabmap_multicam_raw_omni'+(f'_mode{mode}' if mode else '')
 paths={}
 for typ in ('raw','optimized'):
  src=root/f'{stem}_path_{typ}.csv';dst=out/src.name
  if not dst.exists():dst.write_bytes(src.read_bytes())
  with dst.open() as h:rows=list(csv.DictReader(h))
  assert len(rows)==243 and [int(x['node']) for x in rows]==list(range(1,244))
  paths[typ]=np.array([[float(x[k]) for k in ('x','y','z','qx','qy','qz','qw')] for x in rows])
 path=paths['optimized'];raw=paths['raw']
 dst=out/f'{stem}_optimized_all_submaps_xyz.pcd'
 with dst.open('wb') as h:
  header=f'# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {count}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {count}\nDATA binary\n'
  h.write(header.encode())
  for i in range(243):
   cloud=np.load(sub/f'{i:04d}.npy')
   r=Rotation.from_quat(path[i,3:7]);mat=r.as_matrix() if hasattr(r,'as_matrix') else r.as_dcm()
   h.write((cloud@mat.T+path[i,:3]).astype(np.float32).tobytes())
 assert dst.stat().st_size==len(header.encode())+count*12
 sha=hashlib.sha256()
 with dst.open('rb') as h:
  for b in iter(lambda:h.read(4*1024*1024),b''):sha.update(b)
 delta=np.linalg.norm(path[:,:3]-raw[:,:3],axis=1)
 floors={}
 for level in sorted(rank,key=lambda x:rank[x]):
  ids=np.flatnonzero(labels==level)
  if len(ids):
   z=path[ids,2];floors[str(rank[level])]={'nodes':len(ids),'median_z_m':float(np.median(z)),'p90_z_range_m':float(np.percentile(z,95)-np.percentile(z,5))}
 result[str(mode)]={'pcd':dst.name,'bytes':dst.stat().st_size,'points':count,'sha256':sha.hexdigest(),'max_pose_correction_m':float(np.max(delta)),'median_pose_correction_m':float(np.median(delta)),'return_z_m':float(path[-1,2]-path[0,2]),'floor_stats':floors}
 print('MAP',mode,'points',count,'max_correction',round(float(np.max(delta)),3),'return_z',round(float(path[-1,2]-path[0,2]),3),flush=True)
(out/'MAP_RESULTS.json').write_text(json.dumps(result,indent=2))
print('DONE',flush=True)
