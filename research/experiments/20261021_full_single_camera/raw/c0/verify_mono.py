"""Reuse independent four-camera SuperPoint/LightGlue/PnP with symmetric ICP."""
from pathlib import Path
import os
import numpy as np


class Verifier:
    def __init__(self, descriptor_shape, baseline_cache=None):
        import match_reverse as mr
        import pnp_multicam as pn
        import torch
        from common import icp_p2pl, load_submap
        self.mr, self.pn, self.torch = mr, pn, torch
        self.icp, self.load = icp_p2pl, load_submap
        self.views = descriptor_shape[2]
        self.cache = mr.ROOT/'data/featcache'/('v2_'+os.environ.get('SEEKER_VIEW_PROFILE','horizontal3'))
        self.cache.mkdir(parents=True,exist_ok=True)
        self.old = Path(baseline_cache) if baseline_cache else None

    def feats(self, i, c, view):
        mr = self.mr
        key = ('sp',i,c,view)
        if key in mr.cache:
            return mr.cache[key]
        paths = [self.cache/f'sp_{i}_{c}_{view}.npz']
        if self.old and self.views == 3:
            paths.append(self.old/f'sp_{i}_{c}_{view}.npz')
        path = next((p for p in paths if p.exists()),None)
        if path:
            with np.load(path) as z:
                d = {'keypoints':self.torch.from_numpy(z['kp'])[None],
                     'descriptors':self.torch.from_numpy(z['d'])[None],
                     'image_size':self.torch.from_numpy(z['sz'])}
                if 'sc' in z:
                    d['keypoint_scores'] = self.torch.from_numpy(z['sc'])[None]
                value = (z['kp'].copy(),d)
            mr.cache[key] = value
            return value
        value = mr.feats(*key)
        d = value[1]
        fields = dict(kp=d['keypoints'][0].numpy(),d=d['descriptors'][0].numpy(),sz=d['image_size'].numpy())
        if 'keypoint_scores' in d:
            fields['sc'] = d['keypoint_scores'][0].numpy()
        np.savez(paths[0],**fields)
        return value

    def verify(self, i, j, shift, similarity):
        mr = self.mr
        import cv2
        from scipy.spatial.transform import Rotation
        tag = f'mono{self.camera}'
        for f in mr.FEAT.glob(f'{tag}_{i}_{j}_c*_sp.npz'):
            f.unlink()
        per = {}
        for ci in (self.camera,):
            cj = ci
            sub = similarity[ci,:,cj,:]
            top = np.argsort(sub.ravel(),kind='stable')[::-1][:2]
            ba, bb, pixels = [], [], []
            for k in top:
                va,vb = divmod(int(k),self.views)
                a,b = self.feats(i,ci,va),self.feats(j,cj,vb)
                matches = mr.match('sp',a,b)
                if len(matches):
                    ba.append(mr.bearings(a[0][matches[:,0]],va))
                    bb.append(mr.bearings(b[0][matches[:,1]],vb))
                    pixels.extend((va,vb,*a[0][x],*b[0][y]) for x,y in matches)
            if not ba:
                per[ci] = 0
                continue
            ba,bb = np.vstack(ba),np.vstack(bb)
            keep = mr.dedup(ba,bb,pixels)
            ba,bb = ba[keep],bb[keep]
            cv2.setRNGSeed((i*997+j*31+ci)%2147483647)
            count,inliers = mr.ess_ransac(ba,bb)
            per[ci] = count
            np.savez_compressed(mr.FEAT/f'{tag}_{i}_{j}_c{ci}{cj}_sp.npz',ba=ba,bb=bb,inl=inliers,
                                px=np.array([pixels[k] for k in keep],float).reshape(-1,6))
        out = self.pn.solve(i,j,'sp',tag)
        out['e_inliers_per_cam'] = per
        if out.get('T') is None or out.get('inliers',0) < 10:
            return out
        t0 = np.asarray(out['T'])
        source,target = self.load(i),self.load(j)
        refined,fitness,rmse,n,_ = self.icp(source,target,t0)
        delta = np.linalg.inv(t0) @ refined
        # Independent reverse evaluation prevents one-sided overlap scores.
        from common import voxel,normals,transform
        source_grid,target_grid = voxel(source,.2),voxel(target,.2)
        normal,planarity,tree = normals(source_grid)
        backwards = transform(np.linalg.inv(refined),target_grid)
        distance,indices = tree.query(backwards,distance_upper_bound=.3)
        valid = np.isfinite(distance)
        valid[valid] &= planarity[indices[valid]] > .3
        rev_rmse = (float(np.sqrt(np.mean(np.sum((backwards[valid]-source_grid[indices[valid]])
                    *normal[indices[valid]],axis=1)**2))) if valid.any() else None)
        out.update(icp_T=refined.tolist(),icp_fitness=fitness,icp_rmse=rmse,icp_n=n,
                   icp_corr_t=float(np.linalg.norm(delta[:3,3])),
                   icp_corr_r=float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude())),
                   reverse_fitness=float(valid.mean()),reverse_rmse=rev_rmse)
        # Memory is bounded across a five-floor recording; disk cache remains.
        if len(mr.cache)>600:
            mr.cache.clear()
        if len(self.pn._sub)>80:
            self.pn._sub.clear()
        return out


def edge_gate(row,labels,max_odom_disagreement_m=5.,max_odom_disagreement_deg=20.):
    from .floors import floor_relation
    i,j = row['query'],row['match']
    if floor_relation(i,j,labels) != 'same':
        return 'hold_unknown_floor' if floor_relation(i,j,labels)=='unknown' else 'reject_cross_floor'
    o = row.get('sp',{})
    def number(key,default=float('inf')):
        value=o.get(key)
        return float(value) if isinstance(value,(int,float)) and np.isfinite(value) else default
    # Two cameras can jointly constrain the rig pose. The observation still
    # needs symmetric geometry and a later independent travelling sequence.
    if number('inliers',0)<15 or sum(isinstance(v,(int,float)) and np.isfinite(v) and v>=5
                                  for v in o.get('per_cam',{}).values())<1:
        return 'reject_visual_support'
    # Repeated bays can create a long, internally consistent false sequence.
    # Keep it for inspection rather than trusting PCM to discover the alias.
    if not (0<=number('diff_vs_odom_t_m')<=max_odom_disagreement_m
            and 0<=number('diff_vs_odom_r_deg')<=max_odom_disagreement_deg):
        return 'hold_large_pose_disagreement'
    if not (number('icp_fitness',0)>=.25 and number('icp_corr_t')<=.5
            and number('icp_corr_r')<=5 and 0<=number('icp_rmse')<=.18
            and number('reverse_fitness',0)>=.25 and 0<=number('reverse_rmse')<=.18):
        return 'reject_symmetric_geometry'
    transform = np.asarray(o.get('icp_T',[]),float)
    if (transform.shape!=(4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1])
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5)
            or np.linalg.det(transform[:3,:3])<.99):
        return 'reject_invalid_transform'
    return 'pair_pass'
