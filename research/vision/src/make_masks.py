"""Static rig/cable masks: temporal-variance (pixels that never change across 134 keyframes over 400 s)
 + fisheye validity circle + v10 hand-drawn cable polygons. 255 = usable."""
import os as _os; _PRJ = _os.environ.get('SEEKER_RESEARCH_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')))  # research/ root
_DATA = _os.path.join(_os.environ.get('SEEKER_DATA_ROOT', _os.path.join(_PRJ, '..', 'data')), '20261008'); _REPO = _os.path.dirname(_PRJ)
import numpy as np, cv2, json, sys
sys.path.insert(0,_PRJ + '/vision/src'); from seeker import *
out=ROOT/'data/masks'; out.mkdir(parents=True,exist_ok=True)
v10=json.load(open(DATA/'cable_mask_v10_20261009/cable_mask.json'))
models=load_models()
for c in range(4):
    st=np.stack([load_frame(f,c).astype(np.float32) for f in range(0,134)])
    st=(st-st.mean((1,2),keepdims=True))/(st.std((1,2),keepdims=True)+1e-6)
    sd=st.std(0); sd=cv2.GaussianBlur(sd,(0,0),5)
    h,w=sd.shape
    # valid fisheye region: rays with incidence up to ~100deg project inside image
    yy,xx=np.mgrid[:h,:w]
    mean=st.mean(0)
    static=(sd<0.35).astype(np.uint8)*255
    static=cv2.morphologyEx(static,cv2.MORPH_OPEN,np.ones((15,15),np.uint8))
    # keep only big static blobs (rig/carrier), not random walls
    n,lab,stats,_=cv2.connectedComponentsWithStats(static)
    keep=np.zeros_like(static)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>4000: keep[lab==i]=255
    keep=cv2.dilate(keep,np.ones((25,25),np.uint8))
    circ=np.zeros((h,w),np.uint8); cv2.ellipse(circ,(w//2,h//2),(int(w*.47),int(h*.47)),0,0,360,255,-1)
    hand=np.zeros((h,w),np.uint8); prof=v10['cameras'][str(c)]
    px=lambda p:np.rint(np.asarray(p)*[w-1,h-1]).astype(np.int32)
    for poly in prof['polygons']: cv2.fillPoly(hand,[px(poly)],255)
    for l in prof['polylines']: cv2.polylines(hand,[px(l['points'])],False,255,int(round(l['width']*min(h,w))))
    m=((circ>0)&(keep==0)&(hand==0)).astype(np.uint8)*255
    cv2.imwrite(str(out/f'mask_c{c}.png'),m)
    cv2.imwrite(str(out/f'tstd_c{c}.png'),np.clip(sd*255,0,255).astype(np.uint8))
    print(c,'usable frac in circle',(m>0).sum()/(circ>0).sum())
