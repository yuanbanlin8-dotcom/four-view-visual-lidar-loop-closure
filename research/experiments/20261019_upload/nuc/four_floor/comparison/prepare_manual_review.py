from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import json, csv, math, zipfile

base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
root=base/'comparison_20261016'
out=root/'manual_review'
out.mkdir(parents=True,exist_ok=True)
methods={
 'four_joint':base/'sequence_gap_ablation_20261014/gap_24_summary.json',
 'four_independent':root/'four_independent/summary.json',
 **{f'single_c{c}':root/f'single_retrieval/single_c{c}_summary.json' for c in range(4)}
}
accepted={}
for name,path in methods.items():
 s=json.loads(path.read_text())
 edges=s.get('accepted_edges',[])
 accepted[name]={(e['i'],e['j']) for e in edges if e['gnc_weight']>.5}
records={(r['query'],r['match']):r for r in json.loads((base/'loop_latest_20261014/candidates.json').read_text())}
for folder,count in ((root/'four_independent/workers',4),(root/'single_retrieval/workers',8)):
 for k in range(count):
  for r in json.loads((folder/str(k)/'candidates.json').read_text()):
   key=(r['query'],r['match'])
   assert key not in records
   records[key]=r
floors=json.loads((base/'loop_latest_20261014/floor_hypotheses.json').read_text())
rank={x:i+1 for i,x in enumerate(sorted(range(4),key=lambda k:floors['levels'][k]))}
all_edges=set().union(*accepted.values())
def priority(key):
 joint=key in accepted['four_joint']
 others=sum(key in accepted[m] for m in accepted if m!='four_joint')
 floor=rank[floors['labels'][key[0]]]
 return (0 if not joint else 1 if others==0 else 2, -floor, key)
ordered=sorted(all_edges,key=priority)
font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(font_path,18)
small=ImageFont.truetype(font_path,16)
frame_root=base/'vision_input/frames'
tile_w,tile_h=300,310
row_h=420
page_w,page_h=4*tile_w+40,3*row_h+20
audit=[]
for key in ordered:
 i,j=key;r=records[key];sp=r['sp']
 cams=sorted(((int(c),int(n)) for c,n in sp.get('per_cam',{}).items()),key=lambda x:(-x[1],x[0]))
 if len(cams)<2:
  cams += [(c,0) for c in range(4) if c not in {x[0] for x in cams}][:2-len(cams)]
 selected=cams[:2]
 in_methods=[m for m,edges in accepted.items() if key in edges]
 floor=rank[floors['labels'][i]]
 audit.append(dict(i=i,j=j,floor=floor,accepted_methods=in_methods,
                   shift=r.get('shift'),inliers=sp.get('inliers'),per_cam=sp.get('per_cam'),
                   odom_diff_m=sp.get('diff_vs_odom_t_m'),odom_diff_deg=sp.get('diff_vs_odom_r_deg'),
                   icp_fitness=sp.get('icp_fitness'),reverse_fitness=sp.get('reverse_fitness'),
                   icp_rmse=sp.get('icp_rmse'),reverse_rmse=sp.get('reverse_rmse')))
for page_idx in range(math.ceil(len(ordered)/3)):
 page=Image.new('RGB',(page_w,page_h),'white');draw=ImageDraw.Draw(page)
 for slot,key in enumerate(ordered[page_idx*3:(page_idx+1)*3]):
  a=audit[page_idx*3+slot];i,j=key;r=records[key];sp=r['sp']
  y=slot*row_h+10
  shade='#fff2d6' if 'four_joint' not in a['accepted_methods'] else '#e8f6ff' if len(a['accepted_methods'])==1 else '#ecf5e9'
  draw.rectangle([8,y,page_w-8,y+row_h-8],fill=shade,outline='#888888',width=2)
  draw.text((20,y+5),f"F{a['floor']} edge {i}-{j} | accepted: {', '.join(a['accepted_methods'])}",fill='black',font=font)
  draw.text((20,y+30),f"visual {a['inliers']} | ICP fit {a['icp_fitness']:.3f}/{a['reverse_fitness']:.3f} | RMSE {a['icp_rmse']:.3f}/{a['reverse_rmse']:.3f} m | odom {a['odom_diff_m']:.2f} m, {a['odom_diff_deg']:.1f} deg",fill='#222222',font=small)
  cams=sorted(((int(c),int(n)) for c,n in sp.get('per_cam',{}).items()),key=lambda x:(-x[1],x[0]))
  if len(cams)<2:cams += [(c,0) for c in range(4) if c not in {x[0] for x in cams}][:2-len(cams)]
  for pair_idx,(ci,count) in enumerate(cams[:2]):
   cj=(ci+int(r['shift']))%4
   for side,(node,cam,label) in enumerate(((i,ci,'query'),(j,cj,'match'))):
    path=frame_root/f'{node:04d}_c{cam}.png'
    x=20+(pair_idx*2+side)*tile_w
    if path.exists():
     img=Image.open(path).convert('RGB');img.thumbnail((tile_w-10,tile_h-10))
     page.paste(img,(x+(tile_w-10-img.width)//2,y+65+(tile_h-10-img.height)//2))
    else:
     draw.text((x,y+170),'MISSING FRAME',fill='red',font=font)
    draw.text((x+4,y+tile_h+67),f'{label} {node} / cam {cam}' + (f' / inliers {count}' if side==0 else ''),fill='black',font=small)
 page.save(out/f'page_{page_idx+1:02d}.jpg',quality=88)

(out/'edge_audit.json').write_text(json.dumps(audit,indent=2))
with (out/'edge_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(audit[0])+['human_label','human_notes'],extrasaction='ignore');w.writeheader()
 for row in audit:w.writerow({**row,'accepted_methods':';'.join(row['accepted_methods']),
                                'per_cam':json.dumps(row['per_cam'])})
with zipfile.ZipFile(root/'manual_review_pages.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(out.glob('page_*.jpg')):z.write(p,p.name)
 z.write(out/'edge_audit.csv','edge_audit.csv')
print('EDGES',len(ordered),'PAGES',math.ceil(len(ordered)/3),'ZIP',root/'manual_review_pages.zip')
print('BASELINE_ONLY',sum(x not in accepted['four_joint'] for x in ordered))
