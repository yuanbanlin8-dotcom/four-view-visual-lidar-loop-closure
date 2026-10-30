import numpy as np, json
from pathlib import Path
R=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
D=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
cases=[
('raw',R/'frontend_ablation_20261014/legacy_livo_full_0.5x/map_raw.pcd',224.,240.,63.9,0.09,232.,-94.,-0.43,[(190000000,215000000),(225000000,245000000)]),
('full',R/'sequence_gap_ablation_20261014/full_resolution/map_four_floors_20261013_loop24s_corrected_fullres_colored.pcd',7.,23.,9.9,0.28,14.,None,None,[(215000000,245000000),(250000000,280000000)])]
for name,p,y0,y1,x0,slope,yref,z0,zs,passes in cases:
 with p.open('rb') as f:
  while not f.readline().startswith(b'DATA '): pass
  offset=f.tell()
 c=np.memmap(p,dtype=D,mode='r',offset=offset)
 out=[]
 for lo,hi in passes:
  yy=[]; nn=[]
  for start in range(lo,min(hi,len(c)),2000000):
   q=c[start:min(start+2000000,hi,len(c))]
   y=q['y']; k=(y>=y0)&(y<y1)
   if not k.any():continue
   y=y[k];x=q['x'][k];z=q['z'][k]
   n=(x-(x0+slope*(y-yref)))/np.sqrt(1+slope*slope)
   k=(n>-.6)&(n<.6)
   if z0 is None:k&=(z>4)&(z<6.2)
   else:
    dz=z-(z0+zs*(y-yref));k&=(dz>-1.3)&(dz<.7)
   yy.append(y[k].copy());nn.append(n[k].copy())
  out.append((np.concatenate(yy),np.concatenate(nn)))
 widths=[]; gaps=[]; rows=[]
 for a in np.arange(y0,y1,.5):
  two=[]
  for y,n in out:two.append(n[(y>=a)&(y<a+.5)])
  if min(map(len,two))<100:continue
  combined=np.concatenate(two)
  p05,p10,p25,p50,p75,p90,p95=np.percentile(combined,[5,10,25,50,75,90,95])
  width90=(p95-p05)*100
  width80=(p90-p10)*100
  width50=(p75-p25)*100
  gap=abs(np.median(two[0])-np.median(two[1]))*100
  widths.append([width90,width80,width50]);gaps.append(gap)
  rows.append([round(a-y0+.25,2),len(combined),round(width90,2),round(width80,2),round(width50,2),round(gap,2)])
 print(json.dumps({'case':name,'points':[len(t[0]) for t in out],'bins':len(rows),'median_90width_cm':float(np.median(np.array(widths)[:,0])),'median_80width_cm':float(np.median(np.array(widths)[:,1])),'median_50width_cm':float(np.median(np.array(widths)[:,2])),'median_pass_gap_cm':float(np.median(gaps)),'rows_y_count_w90_w80_w50_gap':rows}))
