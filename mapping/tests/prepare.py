from pathlib import Path
import yaml
root=Path('/home/nuc/omni_livo_fisheye_ws')
data=yaml.safe_load(Path('/home/nuc/seeker_ws/calibration/factory_20260910.yaml').read_text())
rows=[]; cfg={}
for i in range(4):
 c=data['cam'+str(i)]
 assert c['camera_model']=='omni' and c['distortion_model']=='radtan'
 xi,fx,fy,cx,cy=c['intrinsics']; d=c['distortion_coeffs']
 assert len(d)>=4 and all(abs(x)<1e-14 for x in d[4:]), 'Unsupported extra distortion terms'
 w,h=c['resolution']
 rows.append('{'+','.join(map(str,[w,h,xi,fx,fy,cx,cy]+d[:4]))+'}')
 cfg['cam_'+str(i)]=dict(cam_model='SeekerOmni',scale=0.5,cam_width=w,cam_height=h,cam_xi=xi,cam_fx=fx,cam_fy=fy,cam_cx=cx,cam_cy=cy,cam_d0=d[0],cam_d1=d[1],cam_d2=d[2],cam_d3=d[3])
(root/'tests/factory_params.h').write_text('const double factory[4][11]={'+',\n'.join(rows)+'};\n')
(root/'src/Omni-LIVO/config/seeker_raw_cam.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
