from measure_z import pcd, RAW, FULL, pass_points, POSES
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

route=POSES[14:21]
order=np.argsort(route[:,2])
fig,axs=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
for row,(map_name,path) in enumerate([('raw',RAW),('corrected',FULL)]):
    cloud=pcd(path)
    for col,(pass_name,frames) in enumerate([('out',(14,20)),('return',(115,120))]):
        p=pass_points(cloud,*frames)
        y=p['y']; x=p['x']; z=p['z']
        center=np.interp(y,route[order,2],route[order,1])
        lateral=x-center
        keep=(y>=-26.5)&(y<-11)&(np.abs(lateral)<4)&(z>-2)&(z<3)
        yy,xx,zz=y[keep],lateral[keep],z[keep]
        hist,bins=np.histogram(zz,bins=np.arange(-2,3.025,.025))
        peaks=np.argsort(hist)[-10:][::-1]
        print(map_name,pass_name,'z peaks',[(round(float(bins[i]),3),int(hist[i])) for i in peaks])
        keep2=(zz>-0.4)&(zz<2.0)
        hist2,bins2=np.histogram(xx[keep2],bins=np.arange(-4,4.025,.025))
        peaks2=np.argsort(hist2)[-14:][::-1]
        print(map_name,pass_name,'lateral peaks',[(round(float(bins2[i]),3),int(hist2[i])) for i in peaks2])
        ax=axs[row,col]
        ax.hist2d(yy[keep2][::15],xx[keep2][::15],bins=(62,160),range=[[-26.5,-11],[-4,4]],cmap='magma',cmin=1)
        ax.set_title(map_name+' '+pass_name)
        ax.set_xlabel('corridor Y (m)')
        ax.set_ylabel('lateral x (m)')
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent/'wall_probe.png',dpi=170)
