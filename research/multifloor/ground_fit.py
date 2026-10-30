"""Deterministic body-frame floor fit with residual and spatial-support checks."""
import numpy as np


def fit_ground(pose, cloud, seed=0):
    up = pose[:3,:3].T @ np.array([0.,0.,1.])
    cloud = np.asarray(cloud, float)
    height = cloud @ up
    band = cloud[(height < -1.2) & (height > -2.6)
                 & (np.linalg.norm(cloud[:,:2],axis=1) < 12)
                 & np.isfinite(cloud).all(axis=1)]
    if len(band) < 200:
        return None
    rng = np.random.default_rng(seed)
    points = band[rng.choice(len(band), min(6000,len(band)), replace=False)]
    best = None
    for _ in range(200):
        sample = points[rng.choice(len(points),3,replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if normal @ up < 0:
            normal = -normal
        if normal @ up < np.cos(np.radians(15)):
            continue
        offset = -normal @ sample[0]
        mask = abs(points @ normal + offset) < .04
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < 150:
        return None
    picked = points[best]
    center = picked.mean(0)
    _, _, axes = np.linalg.svd(picked-center, full_matrices=False)
    normal = axes[-1]
    if normal @ up < 0:
        normal = -normal
    offset = float(-normal @ center)
    if not 1.2 < offset < 2.6:
        return None
    residual = picked @ normal + offset
    spread = np.linalg.eigvalsh(np.cov(picked.T))
    if spread[1] < .08:
        return None  # A narrow stair tread must not certify an entire floor.
    world = pose[:3,:3] @ normal
    return dict(n_b=normal.tolist(),d_b=offset,inliers=int(best.sum()),
                rmse=float(np.sqrt(np.mean(residual**2))),
                z_floor=float(pose[2,3]-offset/world[2]),
                tilt_deg=float(np.degrees(np.arccos(np.clip(world[2],-1,1)))),
                spatial_eigenvalues=spread.tolist())
