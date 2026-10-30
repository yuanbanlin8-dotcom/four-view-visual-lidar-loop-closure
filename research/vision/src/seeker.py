"""Seeker unified (Mei-style, xi>1) camera model, extrinsics, virtual pinhole views, masks."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import json, numpy as np, cv2, yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
ROOT=Path(_os.environ.get('SEEKER_VISION_ROOT', _PRJ + '/vision')); DATA=Path(_DATA)
CAMCFG=Path(_os.environ.get('SEEKER_CAMERA_CONFIG', _REPO + '/mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'))
RUNTIME=Path(_os.environ.get('SEEKER_RUNTIME_CONFIG', str(DATA/'20261008_145008/runtime.yaml')))
KF=Path(_os.environ.get('SEEKER_SUBMAP_ROOT', str(DATA/'independent_20261008_145008_recall_pose_v13')))
CAM_NAMES=['left','right','bright','bleft']

def load_models():
    cfg=yaml.safe_load(CAMCFG.read_text()); out=[]
    for c in range(4):
        p=cfg[f'cam_{c}']
        K=np.array([[p['cam_fx'],0,p['cam_cx']],[0,p['cam_fy'],p['cam_cy']],[0,0,1.]])
        out.append((K,np.array([p['cam_d0'],p['cam_d1'],p['cam_d2'],p['cam_d3']]),float(p['cam_xi'])))
    return out

def load_extrinsics():
    """returns list of (R_cl, t_cl) with p_cam = R_cl p_lidar + t_cl."""
    cams=yaml.safe_load(RUNTIME.read_text())['extrin_calib']['cameras']; d={c['cam_id']:c for c in cams}
    return [(np.array(d[c]['Rcl']).reshape(3,3),np.array(d[c]['Pcl'])) for c in range(4)]

def project(rays,model):
    K,D,xi=model; rays=np.asarray(rays,float)
    n=np.linalg.norm(rays,axis=-1,keepdims=True); rays=rays/np.maximum(n,1e-12)
    den=rays[...,2]+xi; branch=xi if xi<=1 else 1/xi
    valid=np.isfinite(rays).all(-1)&(den>1e-8)&(rays[...,2]>-branch+1e-8)
    x=rays[...,0]/np.where(valid,den,1); y=rays[...,1]/np.where(valid,den,1)
    k1,k2,p1,p2=D; r2=x*x+y*y; rad=1+k1*r2+k2*r2*r2
    u=K[0,0]*(x*rad+2*p1*x*y+p2*(r2+2*x*x))+K[0,2]
    v=K[1,1]*(y*rad+p1*(r2+2*y*y)+2*p2*x*y)+K[1,2]
    return np.where(valid,u,-1).astype(np.float32),np.where(valid,v,-1).astype(np.float32),valid

# virtual pinhole views: (yaw,pitch) in camera frame (x right, y down, z fwd)
VIEWS=[(-50.,0.),(0.,0.),(50.,0.)]
VSIZE=512; VFOV=90.
def view_basis(yaw,pitch):
    yaw,pitch=np.deg2rad([yaw,pitch])
    f=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    r=np.array([np.cos(yaw),0.,-np.sin(yaw)]); d=np.cross(f,r)
    return np.stack([r,d,f],1)   # columns: virtual x,y,z in camera frame => R_cam_view
def view_K(size=VSIZE,fov=VFOV):
    f=size/2/np.tan(np.deg2rad(fov)/2); return np.array([[f,0,(size-1)/2],[0,f,(size-1)/2],[0,0,1.]])
def view_maps(model,yaw,pitch,size=VSIZE,fov=VFOV):
    Kv=view_K(size,fov); R=view_basis(yaw,pitch)
    yy,xx=np.mgrid[:size,:size]; pix=np.stack([xx,yy,np.ones_like(xx)],-1).astype(float)
    rays=(pix@np.linalg.inv(Kv).T)@R.T
    return project(rays,model)

import os
MASK_VERSION=os.environ.get('SEEKER_MASK','v1')   # v1: rig/cable/circle; v2: v1 + ceiling (src/make_masks_v2.py)
def load_mask(c,version=None):
    v=version or MASK_VERSION
    return cv2.imread(str(ROOT/'data/masks'/(f'mask_c{c}.png' if v=='v1' else f'mask2_c{c}.png')),0)

def load_poses():
    P=json.load(open(KF/'poses.json')); out={}
    for p in P:
        M=np.eye(4); M[:3,:3]=Rot.from_quat(p['q']).as_matrix(); M[:3,3]=p['p']; out[p['id']]=M
    return out
def load_frame(fid,c):
    return cv2.imread(str(ROOT/f'data/frames/{fid:04d}_c{c}.png'),0)
