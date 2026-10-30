"""Decode /fisheye/all/compressed at every keyframe time (poses.json), split 4 cams (stacked vertically, each 1088x1280 WxH)."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import json, numpy as np, cv2, bisect
from pathlib import Path
from rosbags.highlevel import AnyReader
DATA=Path(_DATA); OUT=Path(_PRJ + '/vision/data/frames'); OUT.mkdir(parents=True,exist_ok=True)
P=json.load(open(DATA/'independent_20261008_145008_recall_pose_v13/poses.json'))
targets=sorted((p['t'],p['id']) for p in P); tt=[t for t,_ in targets]
best={}
with AnyReader([DATA/'20261008_145008/images_0.bag']) as r:
    con=[c for c in r.connections if c.topic=='/fisheye/all/compressed']
    for c,ts,raw in r.messages(connections=con):
        t=ts*1e-9; k=bisect.bisect_left(tt,t)
        near=[q for q in (k-1,k) if 0<=q<len(tt) and abs(tt[q]-t)<0.3]
        if not near: continue
        m=r.deserialize(raw,c.msgtype); st=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        for q in near:
            d=abs(st-tt[q]); fid=targets[q][1]
            if fid not in best or d<best[fid][0]: best[fid]=(d,st,bytes(m.data))
meta={}
for fid,(d,st,b) in sorted(best.items()):
    img=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_GRAYSCALE)
    h=img.shape[0]//4
    for c in range(4): cv2.imwrite(str(OUT/f'{fid:04d}_c{c}.png'),img[c*h:(c+1)*h])
    meta[fid]=dict(dt=d,stamp=st,shape=list(img.shape))
json.dump(meta,open(OUT/'meta.json','w'),indent=1)
print(len(meta),'frames; max dt',max(v['dt'] for v in meta.values()), next(iter(meta.values())))
