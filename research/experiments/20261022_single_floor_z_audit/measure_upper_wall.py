"""Exploratory full-corridor upper-surface and wall-normal repeatability audit."""
from measure_z import pcd, RAW, FULL, pass_points, POSES
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / 'upper_wall_audit.json'
route = POSES[14:21]
order = np.argsort(route[:,2])


def summarize(values):
    arr=np.array(list(values),dtype=float)
    return {'n':len(arr),'median_abs_m':float(np.median(np.abs(arr))),
            'p75_abs_m':float(np.percentile(np.abs(arr),75)),
            'signed_median_m':float(np.median(arr))}


def extract(points):
    x,y,z=points['x'],points['y'],points['z']
    center=np.interp(y,route[order,2],route[order,1])
    lateral=x-center
    keep=(y>=-26.5)&(y<-11)&(np.abs(lateral)<3)&(z>-2)&(z<2.7)
    y,lateral,z=y[keep],lateral[keep],z[keep]
    station=np.floor((y+26.5)/0.5).astype(np.int16)
    result={'upper':{},'wall':{}}
    # Upper horizontal returns include the wall-ceiling junction. Use a modal
    # 2 cm height to avoid bias from the varying visible wall area.
    for st in range(31):
        q=station==st
        if not np.any(q): continue
        for side,lo,hi in [('left',-2.0,-1.0),('right',1.0,2.0)]:
            v=z[q&(lateral>=lo)&(lateral<hi)&(z>=1.8)&(z<2.6)]
            if len(v)>=100:
                hist,bins=np.histogram(v,bins=np.arange(1.8,2.62,0.02))
                top=np.argmax(hist)
                if hist[top]>=30:
                    peak=(bins[top]+bins[top+1])/2
                    cluster=v[np.abs(v-peak)<0.04]
                    result['upper'][f'{st}_{side}']=float(np.median(cluster))
            # Lateral coordinate of wall mid-height, excluding floor and top.
            a=lateral[q&(lateral>=lo)&(lateral<hi)&(z>=-0.2)&(z<1.4)]
            if len(a)>=100:
                result['wall'][f'{st}_{side}']=float(np.median(a))
    return result


def main():
    out={'corridor_y_m':[-26.5,-11.0], 'station_m':0.5, 'cases':{}}
    for name,path in [('raw',RAW),('corrected',FULL)]:
        cloud=pcd(path)
        outward=extract(pass_points(cloud,14,20))
        returning=extract(pass_points(cloud,115,120))
        out['cases'][name]={}
        for kind in ['upper','wall']:
            common=sorted(outward[kind].keys()&returning[kind].keys())
            out['cases'][name][kind]={k:returning[kind][k]-outward[kind][k] for k in common}
    for kind in ['upper','wall']:
        common=set.intersection(*(set(out['cases'][name][kind]) for name in out['cases']))
        for name in out['cases']:
            rows={k:out['cases'][name][kind][k] for k in sorted(common)}
            out['cases'][name][kind]={'summary':summarize(rows.values()),'rows':rows}
    OUT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({name:{kind:v['summary'] for kind,v in cases.items()}
                      for name,cases in out['cases'].items()},indent=2))


if __name__=='__main__': main()
