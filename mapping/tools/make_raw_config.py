#!/usr/bin/env python3
"""Generate a runtime config only from explicitly supplied installation calibration."""
import argparse
from pathlib import Path
import numpy as np
import yaml

def generate(calibration, output):
    root=Path(__file__).resolve().parents[1]
    data=yaml.safe_load(Path(calibration).read_text())
    ex=data['extrin_calib']; cameras=ex['cameras']
    if len(cameras)!=4 or [c['cam_id'] for c in cameras]!=list(range(4)):
        raise ValueError('Expected ordered physical cameras cam0, cam1, cam2, cam3')
    def transform(R,t):
        R=np.asarray(R,dtype=float); t=np.asarray(t,dtype=float)
        if R.size!=9 or t.size!=3 or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError('Extrinsics must contain finite 3x3 rotation and 3-vector translation in meters')
        R=R.reshape(3,3)
        if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
            raise ValueError('Extrinsic rotation is not a proper rotation')
    for c in cameras: transform(c['Rcl'],c['Pcl'])
    transform(ex['extrinsic_R'],ex['extrinsic_T'])
    offsets=data['time_offset']
    for key in ['imu_time_offset','img_time_offset']:
        if not np.isfinite(float(offsets[key])): raise ValueError('Invalid time offset')
    cfg=yaml.safe_load((root/'src/Omni-LIVO/config/mid360.yaml').read_text())
    factory=yaml.safe_load(Path('/home/nuc/seeker_ws/calibration/factory_20260910.yaml').read_text())
    cfg['extrin_calib']={k:ex[k] for k in ['extrinsic_R','extrinsic_T','cameras']}
    for i,c in enumerate(cfg['extrin_calib']['cameras']): c['img_topic']=factory['cam'+str(i)]['rostopic']
    cfg['time_offset'].update(offsets)
    cfg['vio'].update(inverse_composition_en=False,patch_size=4,patch_pyrimid_level=3)
    cfg['pcd_save'].update(pcd_save_en=False,colmap_output_en=False)
    Path(output).write_text('# Experimental Seeker raw fisheye config; extrinsics supplied externally.\n'+yaml.safe_dump(cfg))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('calibration');p.add_argument('output');a=p.parse_args()
    try: generate(a.calibration,a.output)
    except (ValueError,TypeError,KeyError,OSError) as e: p.exit(2,'Cannot prepare raw config: '+str(e)+'\n')
