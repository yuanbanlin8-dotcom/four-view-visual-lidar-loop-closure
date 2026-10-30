from measure_z import pcd, RAW, FULL, pass_points
import numpy as np

for map_name, path in [('raw', RAW), ('corrected', FULL)]:
    cloud = pcd(path)
    for window_name, frames in [('out',(0,7)),('return',(124,133))]:
        p = pass_points(cloud,*frames)
        x,y,z = p['x'],p['y'],p['z']
        for radius in [1.5,2.5,4.0]:
            keep = (np.abs(x)<radius)&(np.abs(y)<radius)&(z>-3)&(z<1)
            zz=z[keep]
            hist,bins=np.histogram(zz,bins=np.arange(-3,1.025,0.025))
            peaks=np.argsort(hist)[-8:][::-1]
            print(map_name,window_name,radius,len(zz),[(round(float(bins[i]),3),int(hist[i])) for i in peaks])
