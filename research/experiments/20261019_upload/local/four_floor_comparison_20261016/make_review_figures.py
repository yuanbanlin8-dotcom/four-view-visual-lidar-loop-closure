from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).parent
floors=json.loads((root/'floor_hypotheses.json').read_text())
rank={x:i+1 for i,x in enumerate(np.argsort(floors['levels']))}
labels=np.array([rank[x] if x is not None else 0 for x in floors['labels']])
names=['raw','four_joint','four_independent','single_c0','single_c1','single_c2','single_c3']
poses={n:np.loadtxt(root/(n+'_poses.txt')) for n in names}
summaries={n:json.loads((root/(n+'_summary.json')).read_text()) for n in names[1:]}
accepted={n:{(e['i'],e['j']) for e in summaries[n]['accepted_edges'] if e['gnc_weight']>.5}
          for n in names[1:]}
all_edges=set().union(*accepted.values())
palette={'four_independent':'#e69f00','single_c0':'#cc79a7','single_c1':'#009e73',
         'single_c2':'#0072b2','single_c3':'#d55e00'}

fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
for f,ax in enumerate(axes.flat,1):
    ix=np.flatnonzero(labels==f)
    xy=poses['raw'][ix,1:3]
    ax.plot(xy[:,0],xy[:,1],color='#bdbdbd',lw=1.5,zorder=0)
    for key in sorted(all_edges):
        i,j=key
        if labels[i]!=f:continue
        joint=key in accepted['four_joint']
        other=any(key in accepted[n] for n in names[3:]) or key in accepted['four_independent']
        color='#0072b2' if joint and not other else '#009e73' if joint else '#d55e00'
        marker='o' if joint else 'D'
        ax.scatter(*poses['raw'][i,1:3],s=47,marker=marker,color=color,edgecolor='white',linewidth=.7,zorder=3)
        if not joint or f==4:
            ax.annotate(f'{i}-{j}',poses['raw'][i,1:3],xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set_title(f'Floor {f}: joint {sum(labels[i]==f for i,j in accepted["four_joint"])} | '
                 f'independent {sum(labels[i]==f for i,j in accepted["four_independent"])}')
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor='#009e73',label='joint + another method'),
                    Line2D([0],[0],marker='o',color='none',markerfacecolor='#0072b2',label='joint only'),
                    Line2D([0],[0],marker='D',color='none',markerfacecolor='#d55e00',label='other method only')],
           loc='lower center',ncol=3,bbox_to_anchor=(.5,-.025))
fig.savefig(root/'loop_locations.png',dpi=220,bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
fig.subplots_adjust(left=.08,right=.98,top=.93,bottom=.18,hspace=.34,wspace=.22)
for f,ax in enumerate(axes.flat,1):
    ix=np.flatnonzero(labels==f)
    for n in names[2:]:
        d=np.linalg.norm(poses[n][ix,1:4]-poses['four_joint'][ix,1:4],axis=1)
        ax.plot(ix,d,lw=1.6,label=n.replace('_',' '),color=palette[n])
    ax.set_title(f'Floor {f}')
    ax.set_xlabel('Submap index');ax.set_ylabel('Position difference (m)')
    ax.set_ylim(0,.17);ax.grid(alpha=.2)
handles,legend_labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.04),frameon=False)
fig.savefig(root/'pose_difference_vs_joint.png',dpi=220)
plt.close(fig)
print('saved figures')
