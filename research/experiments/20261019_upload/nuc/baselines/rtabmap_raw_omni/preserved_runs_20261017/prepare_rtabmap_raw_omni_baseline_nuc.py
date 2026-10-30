from pathlib import Path
import json,hashlib,numpy as np
base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
sub=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929_submaps')
source=Path('/home/nuc/omnilivo_floor_loop_trial/external_baselines_20261017/four_floor_rtabmap_multicam')
out=source.parent/'four_floor_rtabmap_raw_omni'
out.mkdir(exist_ok=True)
poses=json.loads((sub/'poses.json').read_text())
raw=np.loadtxt(base/'loop_latest_20261014/raw_poses.txt')
assert len(poses)==len(raw)==243
assert max(np.max(np.abs(np.asarray(p['p'])-raw[i,1:4])) for i,p in enumerate(poses))<1e-8
assert max(min(np.linalg.norm(np.asarray(p['q'])-raw[i,4:8]),np.linalg.norm(np.asarray(p['q'])+raw[i,4:8])) for i,p in enumerate(poses))<1e-8
for n in ('rgb','depth','camera_models.txt','camera_local_transforms.txt'):
 dst=out/n
 if not dst.exists():dst.symlink_to(source/n,target_is_directory=n in ('rgb','depth'))
with (out/'odom_poses.txt').open('w') as f:
 for p in poses:
  v=p['p']+p['q']
  f.write(f"{p['id']} {p['t']:.6f} "+' '.join(f'{x:.9f}' for x in v)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'method':'RTAB-Map four-camera LiDAR-depth loop on unoptimized original OmniLIVO frontend poses','input_pose_file':str(sub/'poses.json'),'independent_raw_pose_file':str(base/'loop_latest_20261014/raw_poses.txt'),'pose_sha256':sha(sub/'poses.json'),'raw_pose_sha256':sha(base/'loop_latest_20261014/raw_poses.txt'),'pose_matches_original_raw_txt':True,'local_lidar_submaps':str(sub),'images_and_depth':str(source),'depth_units_per_meter':1000,'nodes':len(poses),'uses_optimized_frontend':False,'uses_our_loop_backend':False}
(out/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
print('READY',out,'NODES',len(poses),'RAW_EQUAL',True,flush=True)
