"""Phase-1 feasibility: lap-2 vs opposite-facing lap-1 cameras on V13 reverse pairs.
Features: SuperPoint+LightGlue, ALIKED+LightGlue, ORB (BF Hamming, ratio .8, mutual).
Per camera: 3 virtual pinhole views (yaw -50/0/+50, 90deg, 512px) with static masks.
Geometry: correspondences pooled per camera pair as unit bearings (camera frame), deduped,
5-point essential RANSAC (thr 0.004 normalized ~ 1px in view) + 8-pt fundamental RANSAC on view pixels for reference."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import sys, json, time, numpy as np, cv2, torch
sys.path.insert(0,_PRJ + '/vision/src'); from seeker import *
from lightglue import SuperPoint, ALIKED, LightGlue
from lightglue.utils import rbd
torch.set_num_threads(8); torch.set_grad_enabled(False)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True); FEAT=ROOT/'data/features'; FEAT.mkdir(parents=True,exist_ok=True)
models=load_models(); ext=load_extrinsics(); poses=load_poses()
MAPS={}
for c in range(4):
    m=load_mask(c)
    for vi,(yw,pt) in enumerate(VIEWS):
        u,v,ok=view_maps(models[c],yw,pt)
        vm=cv2.remap(m,u,v,cv2.INTER_NEAREST,borderValue=0); vm[~ok]=0
        vm=cv2.erode(vm,np.ones((9,9),np.uint8))
        MAPS[c,vi]=(u,v,vm)
Kv=view_K(); Kvi=np.linalg.inv(Kv)
def view_img(fid,c,vi):
    u,v,vm=MAPS[c,vi]; return cv2.remap(load_frame(fid,c),u,v,cv2.INTER_LINEAR), vm
def bearings(kp,vi):
    r=np.c_[kp,np.ones(len(kp))]@Kvi.T@view_basis(*VIEWS[vi]).T
    return r/np.linalg.norm(r,axis=1,keepdims=True)

extr={'sp':SuperPoint(max_num_keypoints=1024).eval(),'aliked':ALIKED(max_num_keypoints=1024).eval()}
lg={'sp':LightGlue(features='superpoint').eval(),'aliked':LightGlue(features='aliked').eval()}
orb=cv2.ORB_create(1500,1.2,8,edgeThreshold=19,fastThreshold=10)
cache={}
def feats(method,fid,c,vi):
    k=(method,fid,c,vi)
    if k in cache: return cache[k]
    img,vm=view_img(fid,c,vi)
    if method=='orb':
        kp,d=orb.detectAndCompute(img,vm)
        r=(np.array([p.pt for p in kp],np.float32).reshape(-1,2),d)
    else:
        t=torch.from_numpy(img).float()[None,None]/255.
        f=extr[method].extract(t)
        kp=f['keypoints'][0].numpy(); keep=vm[np.clip(kp[:,1].round().astype(int),0,VSIZE-1),np.clip(kp[:,0].round().astype(int),0,VSIZE-1)]>0
        f={'keypoints':f['keypoints'][:,keep],'descriptors':f['descriptors'][:,keep],'keypoint_scores':f['keypoint_scores'][:,keep] if 'keypoint_scores' in f else None,'image_size':f['image_size']}
        f={a:b for a,b in f.items() if b is not None}
        r=(f['keypoints'][0].numpy(),f)
    cache[k]=r; return r
bf=cv2.BFMatcher(cv2.NORM_HAMMING)
def match(method,A,B):
    ka,da=A; kb,db=B
    if len(ka)<5 or len(kb)<5: return np.empty((0,2),int)
    if method=='orb':
        if da is None or db is None: return np.empty((0,2),int)
        m1=bf.knnMatch(da,db,2); m2=bf.match(db,da); back={m.queryIdx:m.trainIdx for m in m2}
        out=[(a.queryIdx,a.trainIdx) for a,b in (x for x in m1 if len(x)==2) if a.distance<.8*b.distance and back.get(a.trainIdx)==a.queryIdx]
        return np.array(out,int).reshape(-1,2)
    res=lg[method]({'image0':da,'image1':db}); return rbd(res)['matches'].numpy()

def opposite_combos(i,j):
    """camera combos whose optical axes are within 60deg given lidar-ICP T_ji (used only to pick combos)."""
    Tji=REF[(i,j)]; out=[]
    for ci in range(4):
        zi=Tji[:3,:3]@(ext[ci][0].T@[0,0,1])
        for cj in range(4):
            zj=ext[cj][0].T@[0,0,1]
            ang=np.degrees(np.arccos(np.clip(zi@zj,-1,1)))
            if ang<60: out.append((ci,cj,float(ang)))
    return out

def ess_ransac(ba,bb,thr=0.004):
    # rotate both bearing sets into a mid frame so z>0 for most rays
    m=(ba.mean(0)+bb.mean(0)); m/=np.linalg.norm(m)
    z=m; x=np.cross([0,1,0],z); x/=np.linalg.norm(x); y=np.cross(z,x); Rm=np.stack([x,y,z])
    A=ba@Rm.T; B=bb@Rm.T; ok=(A[:,2]>0.15)&(B[:,2]>0.15)
    idx=np.where(ok)[0]
    if len(idx)<8: return 0,np.zeros(len(ba),bool)
    pa=A[idx,:2]/A[idx,2:]; pb=B[idx,:2]/B[idx,2:]
    best=0; bm=None
    try: E,mask=cv2.findEssentialMat(pa,pb,np.eye(3),cv2.USAC_MAGSAC,0.999,thr)
    except cv2.error: mask=None
    if mask is None: return 0,np.zeros(len(ba),bool)
    inl=np.zeros(len(ba),bool); inl[idx[mask.ravel()>0]]=True
    return int(inl.sum()),inl

def dedup(ba,bb,rows):
    keep=[];seen=[]
    for k in range(len(ba)):
        if any((np.dot(ba[k],sa)>0.99996 and np.dot(bb[k],sb)>0.99996) for sa,sb in seen[-400:]): continue
        seen.append((ba[k],bb[k])); keep.append(k)
    return np.array(keep,int)

def run_pair(i,j,combos,tag):
    rec={'i':i,'j':j,'tag':tag,'combos':[]}
    for ci,cj,ang in combos:
        cr={'ci':ci,'cj':cj,'axis_angle_deg':ang}
        for method in ['sp','aliked','orb']:
            BA=[];BB=[];PX=[];F8=0;raw=0
            for va in range(3):
                for vb in range(3):
                    A=feats(method,i,ci,va); B=feats(method,j,cj,vb); m=match(method,A,B); raw+=len(m)
                    if len(m)>=8:
                        try:
                            Fm,fm=cv2.findFundamentalMat(A[0][m[:,0]],B[0][m[:,1]],cv2.USAC_MAGSAC,1.5,0.999)
                            if fm is not None: F8+=int(fm.sum())
                        except cv2.error: pass
                    if len(m):
                        BA.append(bearings(A[0][m[:,0]],va)); BB.append(bearings(B[0][m[:,1]],vb))
                        PX+= [(va,vb,*A[0][a],*B[0][b]) for a,b in m]
            if BA:
                ba=np.vstack(BA); bb=np.vstack(BB); keep=dedup(ba,bb,PX); ba,bb=ba[keep],bb[keep]; PX=[PX[k] for k in keep]
                n,inl=ess_ransac(ba,bb)
            else: ba=bb=np.empty((0,3)); n=0; inl=np.zeros(0,bool)
            cr[method]={'raw':raw,'dedup':len(ba),'F_inliers_sum_viewpairs':F8,'E_inliers':n}
            np.savez_compressed(FEAT/f'{tag}_{i}_{j}_c{ci}{cj}_{method}.npz',ba=ba,bb=bb,inl=inl,px=np.array(PX,float).reshape(-1,6))
        rec['combos'].append(cr)
    for method in ['sp','aliked','orb']:
        rec[method+'_E_total']=sum(c[method]['E_inliers'] for c in rec['combos'])
        rec[method+'_E_max']=max([c[method]['E_inliers'] for c in rec['combos']]+[0])
    return rec

_reference_path=ROOT/'ref/recall_pose_v13_20261010/results/reverse_route_probe/decisions.json'
D=json.loads(_reference_path.read_text()) if _reference_path.exists() else []
REF={}
for r in D:
    T=np.eye(4); T[:3,:3]=Rot.from_quat([r['qx'],r['qy'],r['qz'],r['qw']]).as_matrix(); T[:3,3]=[r['tx'],r['ty'],r['tz']]; REF[(r['i'],r['j'])]=T
if __name__=='__main__':
    pairs=[(r['i'],r['j']) for r in D]
    results=json.load(open(OUT/'reverse_matches.json')) if (OUT/'reverse_matches.json').exists() else []; t0=time.time()
    done={(r['i'],r['j']) for r in results}
    for (i,j) in pairs:
        if (i,j) in done: continue
        r=run_pair(i,j,opposite_combos(i,j),'rev'); r['geometry_ok']=next(x['geometry_ok'] for x in D if (x['i'],x['j'])==(i,j)); r['v13_orb_inliers']=next(x['visual_total_inliers'] for x in D if (x['i'],x['j'])==(i,j))
        results.append(r); print(i,j,'SP',r['sp_E_total'],'AL',r['aliked_E_total'],'ORB',r['orb_E_total'],f'{time.time()-t0:.0f}s',flush=True)
        json.dump(results,open(OUT/'reverse_matches.json','w'),indent=1)
    # negative controls: same lap-2 frame, lap-1 frame >=15 m away, same camera combos
    P=poses; ctrl=[]
    for (i,j) in pairs[::3]:
        cand=[k for k in range(0,60) if np.linalg.norm(P[k][:3,3]-P[j][:3,3])>15]
        k=min(cand,key=lambda k:abs(k-j)); REF[(i,k)]=REF[(i,j)]
        r=run_pair(i,k,opposite_combos(i,j),'ctrl'); r['true_j']=j; ctrl.append(r)
        print('CTRL',i,k,'SP',r['sp_E_total'],'AL',r['aliked_E_total'],'ORB',r['orb_E_total'],flush=True)
        json.dump(ctrl,open(OUT/'control_matches.json','w'),indent=1)
