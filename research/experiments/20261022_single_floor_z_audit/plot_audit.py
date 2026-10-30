"""Plot the same-route vertical and wall repeatability comparisons."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ground=json.loads((ROOT/'z_audit.json').read_text(encoding='utf-8'))
wall=json.loads((ROOT/'upper_wall_audit.json').read_text(encoding='utf-8'))
blue='#0868ac'; orange='#cf4e23'; ink='#203044'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.labelcolor':ink,'text.color':ink,
                     'axes.edgecolor':ink,'xtick.color':ink,'ytick.color':ink,
                     'axes.linewidth':1.1,'savefig.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(11,7.5),dpi=200)

ax=axs[0,0]
for name,color,label in [('raw',orange,'Before'),('corrected',blue,'After')]:
    rows=ground['regions']['corridor']['cases'][name]['station_signed_median_m']
    ax.plot([r['y_m'] for r in rows],[100*r['signed_gap_m'] for r in rows],
            marker='o',markersize=3.4,lw=2.4,color=color,label=label)
ax.axhline(0,color='#a4afba',lw=1)
ax.set(xlabel='Position along corridor (m)',ylabel='Return − outward height (cm)',
       title='(a) Corridor ground, 15.5 m')
ax.legend(frameon=False,loc='lower left')

for ax,kind,title,ylabel in [
    (axs[0,1],'upper','(b) Upper returns near wall–ceiling junction','Absolute Z gap (cm)'),
    (axs[1,0],'wall','(c) Wall-face lateral repeatability','Absolute lateral gap (cm)')]:
    for name,color,label in [('raw',orange,'Before'),('corrected',blue,'After')]:
        rows=wall['cases'][name][kind]['rows']
        x=np.array([-26.5+(int(key.split('_')[0])+0.5)*0.5 for key in rows])
        y=np.array([abs(value)*100 for value in rows.values()])
        stations=np.unique(x)
        med=np.array([np.median(y[x==s]) for s in stations])
        ax.plot(stations,med,marker='o',markersize=3.4,lw=2.4,color=color,label=label)
    ax.set(xlabel='Position along corridor (m)',ylabel=ylabel,title=title)
    ax.legend(frameon=False)

ax=axs[1,1]
regions=[('classroom','Broad 8 × 8 m'),('classroom_center','Central 3 × 3 m')]
xx=np.arange(2)
before=[100*ground['regions'][r]['cases']['raw']['median_abs_m'] for r,_ in regions]
after=[100*ground['regions'][r]['cases']['corrected']['median_abs_m'] for r,_ in regions]
w=.33
ax.bar(xx-w/2,before,width=w,color=orange,label='Before')
ax.bar(xx+w/2,after,width=w,color=blue,label='After')
for xpos,vals in [(xx-w/2,before),(xx+w/2,after)]:
    for x,y in zip(xpos,vals):
        ax.text(x,y+0.7,f'{y:.1f}',ha='center',va='bottom',fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels([name for _,name in regions])
ax.set_ylim(0,43)
ax.set(ylabel='Median absolute ground gap (cm)',title='(d) Start/finish room ground')
ax.legend(frameon=False)

for ax in axs.flat:
    ax.grid(axis='y',color='#dce2e6',lw=.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_visible(True)
fig.suptitle('Outgoing versus return pass: same-recording point-cloud comparison',
             fontsize=14,fontweight='bold')
fig.tight_layout(rect=(0,0,1,.96))
fig.savefig(ROOT/'single_floor_z_wall_audit.png',dpi=240)
fig.savefig(ROOT/'single_floor_z_wall_audit.pdf')
