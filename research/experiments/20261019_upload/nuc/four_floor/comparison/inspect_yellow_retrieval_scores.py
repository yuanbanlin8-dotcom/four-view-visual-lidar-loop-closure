from pathlib import Path
import json, sys
import numpy as np

base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
release=base/'loop_latest_20261014'
sys.path.insert(0,'/home/nuc/five_floor_optimization_20261013')
from research.multifloor.retrieval import RigRetrieval
from research.multifloor.floors import floor_relation

with np.load(base/'vision_input/global_dinov2.npz') as z:
    desc=z['desc'];nodes=z['nodes']
meta=json.loads((Path(str(base)+'_submaps')/'poses.json').read_text())
times=np.array([m['t'] for m in meta],float)
labels=json.loads((release/'floor_hypotheses.json').read_text())['labels']
rr=RigRetrieval(desc,nodes,times,labels)
accepted=[(e['i'],e['j']) for e in json.loads((base/'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())['accepted_edges'] if e['gnc_weight']>.5]

def selected_pairs(cap):
    pairs=set()
    for q in rr.nodes:
        mean,std=rr.level_stats.get(labels[q],(rr.mean,rr.std))
        rankings=[]
        for k in rr.nodes:
            if (q,k) not in rr.cp or floor_relation(q,k,labels)!='same':continue
            cp=(rr.cp[q,k]-mean)/std
            rankings.append((float(max(np.mean([cp[c,(c+s)%4] for c in range(4)]) for s in range(4))),k))
        bins={};count=0
        for _,k in sorted(rankings,key=lambda x:(-x[0],x[1])):
            b=int((times[k]-times[0])//12)
            if bins.get(b,0)>=cap:continue
            bins[b]=bins.get(b,0)+1
            pairs.add((q,k));count+=1
            if count>=12:break
    return pairs

baseline=selected_pairs(2)
assert len(baseline)==1100
for cap in (3,4):
    alt=selected_pairs(cap)
    print('CAP',cap,'pairs',len(alt),'shared_with_frozen',len(alt&baseline),
          'frozen_accepted_covered',sum(k in alt for k in accepted),'of',len(accepted),
          'yellow_covered',[k for k in [(69,45),(74,41),(78,37),(117,96),(117,97)] if k in alt])

for i,j in [(69,45),(74,41),(78,37),(117,96),(117,97)]:
    mean,std=rr.level_stats.get(labels[i],(rr.mean,rr.std))
    def row(k):
        cp=(rr.cp[i,k]-mean)/std
        shifts=np.array([[cp[c,(c+s)%4] for c in range(4)] for s in range(4)])
        return (float(shifts.mean(1).max()),float(cp.max()),
                int(np.argmax(shifts.mean(1))),shifts[np.argmax(shifts.mean(1))].tolist(),
                np.unravel_index(np.argmax(cp),cp.shape))
    scores={k:row(k) for k in rr.nodes if (i,k) in rr.cp and floor_relation(i,k,labels)=='same'}
    ranked=sorted(scores,key=lambda k:(-scores[k][0],k))
    raw_rank=ranked.index(j)+1
    selected={}
    for cap in (2,3,4):
        chosen=[];bins={}
        for k in ranked:
            b=int((times[k]-times[0])//12)
            if bins.get(b,0)>=cap:continue
            bins[b]=bins.get(b,0)+1
            chosen.append(k)
            if len(chosen)>=12:break
        selected[cap]=chosen
    chosen=selected[2]
    print('EDGE',i,j,'joint_score',round(scores[j][0],3),
          'max_camera_score',round(scores[j][1],3),'joint_raw_rank',raw_rank,
          'joint_diversified_rank',chosen.index(j)+1 if j in chosen else 'not_selected',
          'joint_top12',chosen[:12],
          'cap3_position',selected[3].index(j)+1 if j in selected[3] else None,
          'cap4_position',selected[4].index(j)+1 if j in selected[4] else None,
          'best_shift',scores[j][2],
          'four_cam_scores_at_shift',[round(x,2) for x in scores[j][3]],
          'best_single_pair',scores[j][4])
