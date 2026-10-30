"""Lidar-depth-assisted multi-camera PnP + RANSAC for lap-2 frame i vs lap-1 frame j.
3D: lap-1 keypoint bearings (cam cj) lifted with depth from lidar submap j (keyframe lidar frame).
2D: matched lap-2 bearings (cam ci). Unknown: T_ji (pose of lidar_i in lidar_j), same convention as V13 lidar ICP.
Hypotheses: per-camera P3P/AP3P RANSAC (cv2.solvePnPRansac on rotated tangent plane), scored jointly over all
cameras with angular threshold; best refined with robust (Huber) LM over all cameras' inliers."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import sys, json, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
sys.path.insert(0,_PRJ + '/vision/src'); from seeker import *
FEAT=ROOT/'data/features'; OUT=ROOT/'results'
ext=load_extrinsics(); poses=load_poses()
_reference_path=ROOT/'ref/recall_pose_v13_20261010/results/reverse_route_probe/decisions.json'
D=json.loads(_reference_path.read_text()) if _reference_path.exists() else []
REF={}
for r in D:
    T=np.eye(4); T[:3,:3]=Rot.from_quat([r['qx'],r['qy'],r['qz'],r['qw']]).as_matrix(); T[:3,3]=[r['tx'],r['ty'],r['tz']]; REF[(r['i'],r['j'])]=(T,r['geometry_ok'])
ANG_THR=np.deg2rad(0.8)
_sub={}
def lidar_cam(j,cj):
    k=(j,cj)
    if k not in _sub:
        P=np.load(KF/f'{j:04d}.npy').astype(float); R,t=ext[cj]; pc=P@R.T+t; rng=np.linalg.norm(pc,axis=1)
        ok=(rng>0.4)&(rng<25); b=pc[ok]/rng[ok,None]; _sub[k]=(cKDTree(b),rng[ok])
    return _sub[k]
def lift(bb,j,cj):
    tree,rng=lidar_cam(j,cj); X=np.full((len(bb),3),np.nan)
    nb=tree.query_ball_point(bb,np.deg2rad(0.6))
    for k,ids in enumerate(nb):
        if len(ids)<3: continue
        r=np.sort(rng[ids]); fg=r[:max(3,len(r)//3)]           # foreground cluster
        if fg[-1]-fg[0]>0.08*fg[0]+0.05: continue           # depth discontinuity -> reject
        d=np.median(fg); R,t=ext[cj]; X[k]=R.T@(bb[k]*d-t)
    return X
def Tmat(x): T=np.eye(4); T[:3,:3]=Rot.from_rotvec(x[:3]).as_matrix(); T[:3,3]=x[3:]; return T
def residual_ang(Tji,X,b,cam):
    Tij=np.linalg.inv(Tji); Xi=X@Tij[:3,:3].T+Tij[:3,3]; out=np.empty(len(X))
    for c in np.unique(cam):
        m=cam==c; R,t=ext[c]; xc=Xi[m]@R.T+t; xc/=np.linalg.norm(xc,axis=1,keepdims=True)
        out[m]=np.arccos(np.clip((xc*b[m]).sum(1),-1,1))
    return out
def solve(i,j,method,tag='rev',combo_files=None):
    X=[];B=[];C=[]
    for f in sorted(FEAT.glob(f'{tag}_{i}_{j}_c*_{method}.npz')):
        ci,cj=int(f.name.split('_c')[1][0]),int(f.name.split('_c')[1][1]); z=np.load(f)
        if len(z['ba'])==0: continue
        Xl=lift(z['bb'],j,cj); ok=np.isfinite(Xl).all(1)
        X.append(Xl[ok]); B.append(z['ba'][ok]); C.append(np.full(ok.sum(),ci))
    if not X: return dict(n_corr=0,success=False)
    X=np.vstack(X);B=np.vstack(B);C=np.concatenate(C); best=(0,None)
    for ci in np.unique(C):
        m=C==ci
        if m.sum()<6: continue
        b=B[m]; z=b.mean(0); z/=np.linalg.norm(z); x=np.cross([0,1,0],z); x/=np.linalg.norm(x); y=np.cross(z,x); Rm=np.stack([x,y,z])
        br=b@Rm.T; ok=br[:,2]>0.15
        if ok.sum()<6: continue
        pts=(br[ok,:2]/br[ok,2:]).astype(np.float64); obj=X[m][ok]
        for seed in range(3):
            cv2.setRNGSeed(seed)
            s,rv,tv,inl=cv2.solvePnPRansac(obj,pts,np.eye(3),None,iterationsCount=2000,reprojectionError=np.tan(ANG_THR),confidence=0.999,flags=cv2.SOLVEPNP_AP3P)
            if not s: continue
            Tcj=np.eye(4); Tcj[:3,:3]=Rm.T@cv2.Rodrigues(rv)[0]; Tcj[:3,3]=Rm.T@tv.ravel()     # x_ci = T_ci<-j X_j
            Tci=np.eye(4); Tci[:3,:3],Tci[:3,3]=ext[ci]
            Tji=np.linalg.inv(np.linalg.inv(Tci)@Tcj)
            n=int((residual_ang(Tji,X,B,C)<ANG_THR).sum())
            if n>best[0]: best=(n,Tji)
    if best[1] is None: return dict(n_corr=len(X),success=False,inliers=0)
    T0=best[1]; x0=np.r_[Rot.from_matrix(T0[:3,:3]).as_rotvec(),T0[:3,3]]
    for it in range(2):
        inl=residual_ang(Tmat(x0),X,B,C)<ANG_THR*(1.5 if it==0 else 1)
        if inl.sum()<4: break
        x0=least_squares(lambda x:residual_ang(Tmat(x),X[inl],B[inl],C[inl]),x0,loss='huber',f_scale=ANG_THR/2).x
    T=Tmat(x0); res=residual_ang(T,X,B,C); inl=res<ANG_THR
    cams=sorted(set(C[inl].tolist())); per={int(c):int((inl&(C==c)).sum()) for c in np.unique(C)}
    out=dict(n_corr=int(len(X)),inliers=int(inl.sum()),inlier_cams=len([c for c in per if per[c]>=5]),per_cam=per,
             median_err_deg=float(np.degrees(np.median(res[inl]))) if inl.any() else None,T=T.tolist())
    out['success']=bool(out['inliers']>=20 and out['inlier_cams']>=2)
    if (i,j) in REF:
        Tr=REF[(i,j)][0]; dT=np.linalg.inv(Tr)@T
        out['err_vs_lidar_icp_t_m']=float(np.linalg.norm(dT[:3,3])); out['err_vs_lidar_icp_r_deg']=float(np.degrees(np.linalg.norm(Rot.from_matrix(dT[:3,:3]).as_rotvec())))
        out['ref_geometry_ok']=REF[(i,j)][1]
    To=np.linalg.inv(poses[j])@poses[i]; dT=np.linalg.inv(To)@T
    out['diff_vs_odom_t_m']=float(np.linalg.norm(dT[:3,3])); out['diff_vs_odom_r_deg']=float(np.degrees(np.linalg.norm(Rot.from_matrix(dT[:3,:3]).as_rotvec())))
    dI=np.linalg.inv(To)@REF.get((i,j),(To,0))[0]
    out['icp_vs_odom_t_m']=float(np.linalg.norm(dI[:3,3]))
    return out
if __name__=='__main__':
    rev=json.load(open(OUT/'reverse_matches.json')); res=[]
    ctrl=json.load(open(OUT/'control_matches.json')) if (OUT/'control_matches.json').exists() else []
    for tag,lst in [('rev',rev),('ctrl',ctrl)]:
        for r in lst:
            for m in ['sp','aliked','orb']:
                o=solve(r['i'],r['j'],m,tag); o.update(i=r['i'],j=r['j'],method=m,tag=tag); res.append(o)
                print(tag,r['i'],r['j'],m,o.get('inliers'),o.get('inlier_cams'),o['success'],round(o.get('err_vs_lidar_icp_t_m',-1),3),round(o.get('err_vs_lidar_icp_r_deg',-1),2),flush=True)
    json.dump(res,open(OUT/'pnp_results.json','w'),indent=1)
