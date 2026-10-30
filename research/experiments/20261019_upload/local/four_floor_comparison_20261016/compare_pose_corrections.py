from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

root=Path(__file__).parent
floors=json.loads((root/'floor_hypotheses.json').read_text())
rank={x:i+1 for i,x in enumerate(np.argsort(floors['levels']))}
labels=np.array([rank[x] if x is not None else 0 for x in floors['labels']])
names=['raw','four_joint','four_independent','single_c0','single_c1','single_c2','single_c3']
poses={n:np.loadtxt(root/(n+'_poses.txt')) for n in names}
assert all(x.shape==(243,8) for x in poses.values())
assert all(np.allclose(poses[n][:,0],poses['raw'][:,0]) for n in names)

def summarize(a,b):
    diff=b[:,1:4]-a[:,1:4]
    d3=np.linalg.norm(diff,axis=1)
    dz=np.abs(diff[:,2])
    aa=Rotation.from_quat(a[:,4:8]);bb=Rotation.from_quat(b[:,4:8])
    angle=np.degrees((aa.inv()*bb).magnitude())
    out={}
    for f in range(1,5):
        ix=labels==f
        out[str(f)]={'nodes':int(ix.sum()),
                     'median_3d_m':float(np.median(d3[ix])),
                     'p95_3d_m':float(np.percentile(d3[ix],95)),
                     'max_3d_m':float(d3[ix].max()),
                     'max_vertical_m':float(dz[ix].max()),
                     'max_rotation_deg':float(angle[ix].max())}
    out['all']={'median_3d_m':float(np.median(d3)),
                'p95_3d_m':float(np.percentile(d3,95)),
                'max_3d_m':float(d3.max()),
                'max_vertical_m':float(dz.max()),
                'max_rotation_deg':float(angle.max())}
    return out

report={'relative_to_raw':{},'relative_to_four_joint':{}}
for n in names[1:]:
    report['relative_to_raw'][n]=summarize(poses['raw'],poses[n])
for n in names[2:]:
    report['relative_to_four_joint'][n]=summarize(poses['four_joint'],poses[n])
(root/'pose_correction_comparison.json').write_text(json.dumps(report,indent=2))
for category,group in report.items():
    print(category)
    for name,d in group.items():
        a=d['all']
        print(name,'median_m',round(a['median_3d_m'],3),
              'p95_m',round(a['p95_3d_m'],3),'max_m',round(a['max_3d_m'],3),
              'max_z_m',round(a['max_vertical_m'],4),
              'max_rot_deg',round(a['max_rotation_deg'],2),
              'floor4_max_m',round(d['4']['max_3d_m'],3))
