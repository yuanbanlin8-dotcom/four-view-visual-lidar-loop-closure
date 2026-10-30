"""Shared helpers for the Seeker Omni-D ground-truth-free evaluation."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import json, os, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

DATA = _DATA
RUN = os.path.join(DATA, '20261008_145008')
SUBMAPS = os.environ.get('SEEKER_SUBMAP_ROOT', os.path.join(DATA, 'independent_20261008_145008_recall_pose_v13'))
EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_SETS = {
    # name: (file, description)
    'raw_odometry': (os.path.join(SUBMAPS, 'keyframe_poses.txt'),
                     'Omni-LIVO front-end keyframe poses (no loop closure)'),
    'v12v13_5edge': (os.path.join(DATA, 'recall_pose_v13_20261010/results/combined/event_fusion_poses.txt'),
                     'V12/V13 combined GTSAM result, 5 loop edges 130-0,96-39,94-41,95-40,119-15'),
    'backend_51edge': (_PRJ + '/backend/results/ablation/poses_loops5_floor_gicp.txt',
                       'Backend full graph C: odometry + 5 V13 edges + 46 reverse GICP loops + floor/gravity factor'),
}
EDGES = {'v12v13_5edge': os.path.join(DATA, 'recall_pose_v13_20261010/results/combined/event_fusion.edges'),
         'backend_51edge': _PRJ + '/backend/results/ablation/edges_loops5_gicp.edges'}
# Nature-style vivid palette (no navy / blue-grey)
PAL = {'raw_odometry': '#E64B35', 'v12v13_5edge': '#00A087', 'backend_51edge': '#8E44AD', 'c3': '#F39B7F', 'c4': '#4DBBD5',
       'c5': '#FFB000', 'c6': '#DC0000', 'c7': '#7E6148', 'c8': '#9C27B0'}

def load_poses(path):
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1:4], a[:, 4:8]  # t, p, q (xyzw)

def T_of(p, q):
    T = np.eye(4); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p; return T

def load_submap(i):
    return np.load(os.path.join(SUBMAPS, '%04d.npy' % i)).astype(np.float64)

def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]

def voxel(pts, s):
    k = np.floor(pts / s).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return pts[np.sort(idx)]

def normals(pts, k=10):
    tree = cKDTree(pts); _, nn = tree.query(pts, k=k)
    nb = pts[nn]; c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c) / k
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]; planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return n, planarity, tree

def icp_p2pl(src, dst, T0, iters=30, max_d=1.0, final_d=0.3):
    """Point-to-plane ICP (src->dst). Returns T, fitness (inlier frac @final_d), rmse, n_inliers."""
    src = voxel(src, 0.2); dst = voxel(dst, 0.2)
    n, pl, tree = normals(dst)
    T = T0.copy()
    for it in range(iters):
        d_thr = max_d + (final_d - max_d) * min(1, it / (iters * 0.6))
        s = transform(T, src); d, j = tree.query(s, distance_upper_bound=d_thr)
        m = np.isfinite(d) & (pl[np.minimum(j, len(dst) - 1)] > 0.3)
        if m.sum() < 50: break
        s_, q_, n_ = s[m], dst[j[m]], n[j[m]]
        r = np.einsum('ij,ij->i', s_ - q_, n_)
        A = np.hstack([np.cross(s_, n_), n_])
        w = 1.0 / (1 + (r / 0.1) ** 2)
        x = np.linalg.lstsq(A * w[:, None] ** .5, -r * w ** .5, rcond=None)[0]
        dT = np.eye(4); dT[:3, :3] = R.from_rotvec(x[:3]).as_matrix(); dT[:3, 3] = x[3:]
        T = dT @ T
        if np.linalg.norm(x) < 1e-5: break
    s = transform(T, src); d, j = tree.query(s, distance_upper_bound=final_d)
    m = np.isfinite(d)
    rmse = float(np.sqrt(np.mean(np.einsum('ij,ij->i', s[m] - dst[j[m]], n[j[m]]) ** 2))) if m.any() else float('nan')
    # degeneracy: smallest eigenvalue of translational Hessian (normalised)
    if m.sum() > 10:
        H = n[j[m]].T @ n[j[m]] / m.sum(); ev = np.linalg.eigvalsh(H)
    else: ev = np.array([np.nan] * 3)
    return T, float(m.mean()), rmse, int(m.sum()), ev

def lap_split(t, p):
    """Lap boundary = keyframe in the middle 30-70% of time closest (xy) to the start."""
    n = len(t); lo, hi = int(.3 * n), int(.7 * n)
    d = np.linalg.norm(p[lo:hi, :2] - p[0, :2], axis=1)
    return lo + int(np.argmin(d))

def yaw_of(q):
    """Heading of the body x axis projected on the world xy plane (deg)."""
    x = R.from_quat(q).apply([1, 0, 0]); return np.degrees(np.arctan2(x[..., 1], x[..., 0]))
