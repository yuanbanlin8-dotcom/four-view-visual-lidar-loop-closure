"""Make a four-camera, LiDAR-depth RTAB-Map input with FAST-LIO2 odometry."""
from pathlib import Path
import csv, json, os, sys
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

os.environ.update(SEEKER_RESEARCH_ROOT='/home/nuc/five_floor_optimization_20261013/research', SEEKER_SUBMAP_ROOT='/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929_submaps', SEEKER_RUNTIME_CONFIG='/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929/loop_inputs/runtime.yaml', SEEKER_CAMERA_CONFIG='/home/nuc/five_floor_optimization_20261013/mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml', SEEKER_VISION_ROOT='/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929/vision', SEEKER_MASK='v2')
sys.path.insert(0, '/home/nuc/five_floor_optimization_20261013/research/vision/src')
from seeker import load_models, load_extrinsics, load_frame, load_mask, view_maps, view_K, view_basis

root = Path('/home/nuc/omnilivo_floor_loop_trial/external_baselines_20261017')
sub = Path(os.environ['SEEKER_SUBMAP_ROOT'])
out = root/'four_floor_rtabmap_multicam'
rgbdir = out/'rgb'; depdir = out/'depth'
rgbdir.mkdir(parents=True,exist_ok=True); depdir.mkdir(parents=True,exist_ok=True)
poses = json.loads((sub/'poses.json').read_text())
odom = next((root/'four_floor_fastlio_btc').glob('*_fastlio_odom.csv'))
with odom.open() as f: rows = list(csv.DictReader(f))
ts = np.array([float(x['field.header.stamp'])/1e9 for x in rows])
model = load_models(); extr = load_extrinsics(); basis = view_basis(0,0)
size = 384; scale = size/512
K = view_K().copy(); K[:2,:] *= scale
cal=[]
for c in range(4):
 um,vm,valid=view_maps(model[c],0,0)
 mask=load_mask(c)
 viewmask=cv2.remap(mask,um,vm,cv2.INTER_NEAREST,borderValue=0)
 viewmask[~valid]=0
 viewmask=cv2.resize(viewmask,(size,size),interpolation=cv2.INTER_NEAREST)>0
 Rcl,Pcl=extr[c]; A=basis.T@Rcl; b=basis.T@Pcl
 T=np.eye(4);T[:3,:3]=A.T;T[:3,3]=-A.T@b
 cal.append((um,vm,viewmask,Rcl,Pcl,T))
np.savetxt(out/'camera_models.txt',np.array([K[0,0],K[1,1],K[0,2],K[1,2],size],dtype=float).reshape(1,-1),fmt='%.9f')
with (out/'camera_local_transforms.txt').open('w') as f:
 for _,_,_,_,_,T in cal:f.write(' '.join(f'{x:.9f}' for x in T[:3,:].reshape(-1))+'\n')

stats=[]
with (out/'odom_poses.txt').open('w') as f:
 for p in poses:
  i=int(p['id']);stamp=float(p['t'])
  j=np.searchsorted(ts,stamp); j=min(max(j,1),len(ts)-1)
  if abs(ts[j-1]-stamp)<abs(ts[j]-stamp):j-=1
  row=rows[j]
  xyz=[float(row['field.pose.pose.position.'+a]) for a in 'xyz']
  q=[float(row['field.pose.pose.orientation.'+a]) for a in 'xyzw']
  f.write(f'{i} {stamp:.6f} '+' '.join(f'{v:.9f}' for v in xyz+q)+'\n')
  cloud=np.load(sub/f'{i:04d}.npy')
  rgbs=[];deps=[];counts=[]
  for c,(um,vm,viewmask,Rcl,Pcl,T) in enumerate(cal):
   image=load_frame(i,c)
   if image is None:raise RuntimeError(f'missing camera {c} node {i}')
   rect=cv2.remap(image,um,vm,cv2.INTER_LINEAR,borderValue=127)
   rect=cv2.resize(rect,(size,size),interpolation=cv2.INTER_AREA)
   if rect.ndim==3:rect=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY)
   rect[~viewmask]=127
   pv=(cloud@Rcl.T+Pcl)@basis
   good=(pv[:,2]>.2)&(pv[:,2]<30)
   proj=pv[good]
   u=np.rint(K[0,0]*proj[:,0]/proj[:,2]+K[0,2]).astype(np.int32)
   v=np.rint(K[1,1]*proj[:,1]/proj[:,2]+K[1,2]).astype(np.int32)
   inside=(u>=0)&(u<size)&(v>=0)&(v<size)
   sparse=np.full((size,size),np.inf,np.float32)
   np.minimum.at(sparse,(v[inside],u[inside]),proj[:,2][inside])
   holes=~np.isfinite(sparse)
   distance,nearest=distance_transform_edt(holes,return_indices=True)
   depth=sparse[nearest[0],nearest[1]]
   valid_depth=(distance<=2)&viewmask&np.isfinite(depth)&(depth<30)
   d16=np.zeros((size,size),np.uint16)
   d16[valid_depth]=np.rint(depth[valid_depth]*1000).astype(np.uint16)
   rgbs.append(rect);deps.append(d16);counts.append(int(np.count_nonzero(d16)))
  assert cv2.imwrite(str(rgbdir/f'{i:04d}.png'),np.hstack(rgbs))
  assert cv2.imwrite(str(depdir/f'{i:04d}.png'),np.hstack(deps))
  stats.append({'node':i,'odom_time_offset_s':float(ts[j]-stamp),'depth_pixels':counts})
  if i%10==0:print('PREP',i,'depth',counts,'odom_dt',round(ts[j]-stamp,3),flush=True)
manifest={'method':'4 synchronized fisheye cameras rectified to virtual pinhole views; LiDAR submap projected to 16-bit depth in millimeters; FAST-LIO2 odometry','nodes':len(stats),'camera_count':4,'per_camera_size':[size,size],'depth_units_per_meter':1000,'odom_file':str(odom),'max_abs_odom_dt_s':max(abs(x['odom_time_offset_s']) for x in stats),'frames':stats}
(out/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE',len(stats),'max_odom_dt_s',manifest['max_abs_odom_dt_s'],flush=True)
