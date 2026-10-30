"""ICP-Hessian degeneracy analysis for loop edges (round 3).

For an edge (i,j) with refined pose T_ji (pose of lidar_i in lidar_j), re-evaluate point-to-plane residuals of
submap i (0.2 m voxel) against submap j at T_ji (0.3 m gate, planarity>0.3, same as eval/scripts/common.icp_p2pl).
Hessian (information, left-perturbation [rot; trans] in frame j):  H = sum_k J_k^T J_k / sigma^2,
   J_k = [ (s_k x n_k)^T , n_k^T ],  sigma = max(rmse, SIGMA_MIN).
Covariance = H^-1 + floor (SIGMA_FLOOR_T, SIGMA_FLOOR_R) to account for unmodelled errors (calibration ~2 deg, submap
motion distortion); ICP covariance alone is known to be over-confident.
Degeneracy uses the *normalised* translational block Ht/N = mean(n n^T) (scale-free, like eval's icp_min_eig):
   degenerate if  min_eig(Ht/N) < MIN_EIG  or  cond(Ht/N) > MAX_COND.
The weak direction is reported in the lidar_j frame and in the map (raw-odometry world) frame.
Backend use: pass `info_6x6` (inverse of cov) to down-weight the weak axis instead of discarding the edge."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import sys, numpy as np
sys.path.insert(0,_PRJ + '/eval/scripts')
from common import voxel, normals, transform, load_submap
SIGMA_MIN=0.02; SIGMA_FLOOR_T=0.10; SIGMA_FLOOR_R=np.radians(1.0); MIN_EIG=0.03; MAX_COND=25.
_cache={}
def _dst(j):
    if j not in _cache:
        d=voxel(load_submap(j),0.2); n,pl,tree=normals(d); _cache[j]=(d,n,pl,tree)
    return _cache[j]
def analyse(i,j,T_ji,R_world_j=None,final_d=0.3):
    src=voxel(load_submap(i),0.2); d,n,pl,tree=_dst(j)
    s=transform(T_ji,src); dist,idx=tree.query(s,distance_upper_bound=final_d)
    m=np.isfinite(dist); m[m]&=pl[idx[m]]>0.3
    s_,q_,n_=s[m],d[idx[m]],n[idx[m]]; N=int(m.sum())
    if N<30: return dict(n_corr=N,degenerate=True,min_eig=0.,cond=np.inf,weak_dir_j=[np.nan]*3,weak_dir_map=[np.nan]*3,cov_6x6=None,info_6x6=None)
    r=np.einsum('ij,ij->i',s_-q_,n_); sigma=max(float(np.sqrt(np.mean(r**2))),SIGMA_MIN)
    J=np.hstack([np.cross(s_,n_),n_]); H=J.T@J/sigma**2
    Ht=n_.T@n_/N; w,V=np.linalg.eigh(Ht); weak=V[:,0]
    cov=np.linalg.inv(H+1e-9*np.eye(6))+np.diag([SIGMA_FLOOR_R**2]*3+[SIGMA_FLOOR_T**2]*3)
    wm=(R_world_j@weak) if R_world_j is not None else weak
    return dict(n_corr=N,rmse=sigma,min_eig=float(w[0]),eigs=w.tolist(),cond=float(w[2]/max(w[0],1e-12)),
                weak_dir_j=weak.tolist(),weak_dir_map=wm.tolist(),degenerate=bool(w[0]<MIN_EIG or w[2]/max(w[0],1e-12)>MAX_COND),
                weak_axis_sigma_m=float(np.sqrt(weak@cov[3:,3:]@weak)),cov_6x6=cov.tolist(),info_6x6=np.linalg.inv(cov).tolist())
