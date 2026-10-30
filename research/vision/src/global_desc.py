"""DINOv2 ViT-B/14 global descriptors on rig-masked rectified views (12 views/frame = 4 cams x 3 yaws).
Descriptor = L2norm([CLS ; GeM(p=3) over masked patch tokens]). Saved: data/global_dinov2.npz  desc[node,cam,view,D]."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import sys, numpy as np, cv2, torch
sys.path.insert(0,_PRJ + '/vision/src'); from seeker import *
torch.set_num_threads(8); torch.set_grad_enabled(False)
NODES=list(range(10,122)); S=336
models=load_models(); M={}
for c in range(4):
    m=load_mask(c)
    for v,(yw,pt) in enumerate(VIEWS):
        u,vv,ok=view_maps(models[c],yw,pt); vm=cv2.remap(m,u,vv,cv2.INTER_NEAREST); vm[~ok]=0; M[c,v]=(u,vv,vm)
net=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14').eval()
mean=np.array([0.485,0.456,0.406]);std=np.array([0.229,0.224,0.225])
out=np.zeros((134,4,3,1536),np.float32)
for n in NODES:
    batch=[];pm=[]
    for c in range(4):
        im=load_frame(n,c)
        for v in range(3):
            u,vv,vm=M[c,v]; x=cv2.remap(im,u,vv,cv2.INTER_LINEAR); x[vm==0]=int(x[vm>0].mean()) if (vm>0).any() else 0
            x=cv2.resize(x,(S,S),interpolation=cv2.INTER_AREA).astype(np.float32)/255.
            batch.append((np.repeat(x[...,None],3,-1)-mean)/std)
            pm.append(cv2.resize(vm,(S//14,S//14),interpolation=cv2.INTER_AREA)>127)
    t=torch.from_numpy(np.stack(batch).transpose(0,3,1,2)).float()
    f=net.forward_features(t); cls=f['x_norm_clstoken'].numpy(); pt=f['x_norm_patchtokens'].numpy()
    for k in range(12):
        p=pt[k][pm[k].ravel()] if pm[k].sum()>10 else pt[k]
        p=p/np.linalg.norm(p,axis=1,keepdims=True)
        g=np.mean(np.clip(p,1e-6,None)**3,0)**(1/3); g/=np.linalg.norm(g); cl=cls[k]/np.linalg.norm(cls[k])
        d=np.r_[cl,g]; out[n,k//3,k%3]=d/np.linalg.norm(d)
    if n%10==0: print(n,flush=True)
np.savez_compressed(ROOT/('data/global_dinov2.npz' if MASK_VERSION=='v1' else f'data/global_dinov2_{MASK_VERSION}.npz'),desc=out,nodes=np.array(NODES))
