from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

root=Path(__file__).resolve().parent
labels=json.loads((root/'floor_hypotheses.json').read_text())['labels']
measurements=json.loads((root/'ground_measurements.json').read_text())
summary={name:json.loads((root/f'{name}_summary.json').read_text()) for name in ('full_reproduction','no_ground_factors')}
colors={0:'#1f77b4',1:'#ff7f0e',2:'#2ca02c',3:'#d62728'}

def observed(name):
 table=np.loadtxt(root/f'{name}_poses.txt')
 rotations=Rotation.from_quat(table[:,4:8]).as_matrix()
 z=np.full(len(table),np.nan)
 for i,m in enumerate(measurements):
  if m is None or labels[i] is None:continue
  normal=rotations[i]@np.asarray(m['n_b'])
  if abs(normal[2])>.2:z[i]=table[i,3]-m['d_b']/normal[2]
 return z

series={name:observed(name) for name in summary}
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,constrained_layout=True)
for ax,(name,title) in zip(axes,(('full_reproduction','Full method'),('no_ground_factors','Without ground factors'))):
 for level in sorted(colors):
  ids=np.array([i for i,l in enumerate(labels) if l==level and np.isfinite(series[name][i])])
  if not len(ids):continue
  center=np.median(series[name][ids])
  ax.scatter(ids,series[name][ids]-center,s=11,color=colors[level],alpha=.8,label=f'Floor {level+1}')
 ax.axhline(0,color='#444444',linewidth=.7)
 ax.set_title(title)
 ax.set_xlabel('Node index along route')
 ax.grid(alpha=.18)
axes[0].set_ylabel('Observed ground height − floor median (m)')
axes[0].legend(ncol=2,fontsize=8,loc='upper center')
axes[0].set_ylim(-.75,.75)
fig.savefig(root/'ground_factor_ablation_profiles.png',dpi=200)

for name,z in series.items():
 print(name)
 for level in sorted(colors):
  ids=np.array([i for i,l in enumerate(labels) if l==level and np.isfinite(z[i])])
  vals=z[ids]
  if len(vals):print(level,len(vals),'p90',np.percentile(vals,95)-np.percentile(vals,5),'std',np.std(vals))
 start=[i for i in range(min(30,len(z))) if labels[i]==0 and np.isfinite(z[i])]
 end=[i for i in range(max(0,len(z)-30),len(z)) if labels[i]==0 and np.isfinite(z[i])]
 if start and end:print('first_floor_start_end_ground_median_gap_m',np.median(z[end])-np.median(z[start]),'samples',len(start),len(end))
