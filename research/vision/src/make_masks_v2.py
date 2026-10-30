"""Mask v2 = mask v1 (static rig/cable + fisheye circle) AND NOT ceiling.
Ceiling = pixels whose viewing ray, expressed in the gravity-aligned frame, has elevation > CEIL_ELEV_DEG.
Gravity direction in the lidar frame = mean over carried nodes (10-121) of R_wl^T z_w (rig tilt is ~constant while carried).
The repetitive grid ceiling is the main suspected source of perceptual aliasing in the ungated test."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import sys, numpy as np, cv2
sys.path.insert(0,_PRJ + '/vision/src'); from seeker import *
CEIL_ELEV_DEG=25.
P=load_poses(); ups=np.array([P[k][:3,:3].T@[0,0,1] for k in range(10,122)]); up=ups.mean(0); up/=np.linalg.norm(up)
spread=np.degrees(np.arccos(np.clip(ups@up,-1,1)))
print('up in lidar frame',up.round(3),'spread deg median/p95',np.median(spread).round(2),np.percentile(spread,95).round(2))
models=load_models(); ext=load_extrinsics(); out=ROOT/'data/masks'
def unproject_grid(model,h,w,step=1):
    # numerically invert project(): sample sphere densely, splat elevation via remap lookup
    K,D,xi=model
    th=np.radians(np.linspace(0,110,900)); ph=np.radians(np.linspace(-180,180,1800))
    T,Ph=np.meshgrid(th,ph,indexing='ij'); rays=np.stack([np.sin(T)*np.cos(Ph),np.sin(T)*np.sin(Ph),np.cos(T)],-1).reshape(-1,3)
    u,v,ok=project(rays,model); return rays[ok],u[ok],v[ok]
tiles=[]
for c in range(4):
    m1=load_mask(c); h,w=m1.shape
    rays,u,v=unproject_grid(models[c],h,w)
    R,_=ext[c]; el=np.degrees(np.arcsin(np.clip((rays@R)@up,-1,1)))   # ray_lidar = R^T ray_cam -> (R^T r).up = r.(R up)... use rays@R == (R^T r)^T
    ceil=np.zeros((h,w),np.uint8); cnt=np.zeros((h,w),np.float32); acc=np.zeros((h,w),np.float32)
    ui=np.clip(u.round().astype(int),0,w-1); vi=np.clip(v.round().astype(int),0,h-1)
    np.add.at(acc,(vi,ui),el); np.add.at(cnt,(vi,ui),1)
    elmap=np.where(cnt>0,acc/np.maximum(cnt,1),np.nan)
    # fill holes
    filled=cv2.inpaint(np.nan_to_num(elmap,nan=0).astype(np.float32),(cnt==0).astype(np.uint8),3,cv2.INPAINT_NS)
    ceil=(filled>CEIL_ELEV_DEG).astype(np.uint8)*255
    m2=((m1>0)&(ceil==0)).astype(np.uint8)*255
    cv2.imwrite(str(out/f'mask2_c{c}.png'),m2)
    print(c,'usable v1',(m1>0).mean().round(3),'v2',(m2>0).mean().round(3))
    im=cv2.cvtColor(load_frame(100,c),cv2.COLOR_GRAY2BGR); ov=im.copy()
    ov[m1==0]=(0.4*ov[m1==0]+0.6*np.array([60,20,255])).astype(np.uint8)
    sel=(m1>0)&(ceil>0); ov[sel]=(0.4*ov[sel]+0.6*np.array([255,200,0])).astype(np.uint8)
    tiles.append(cv2.resize(ov,None,fx=.35,fy=.35))
cv2.imwrite(str(ROOT/'figures/masks_v2_overlay.jpg'),np.hstack(tiles))
