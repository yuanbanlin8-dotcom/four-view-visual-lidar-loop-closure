from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,csv,math,zipfile

base=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
root=base/'comparison_20261016/no_icp_gate'
summary=json.loads((root/'summary.json').read_text())
records={(r['query'],r['match']):r for r in json.loads((base/'loop_latest_20261014/candidates.json').read_text())}
frames=base/'vision_input/frames'
entries=summary['visual_only_pair_pass']
anchors={(r['i'],r['j']) for r in summary['visual_anchor_edges']}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
W,H=1240,1280;row_h=420;tile_w,tile_h=300,300
audit=[]
for e in entries:
    key=(e['i'],e['j']);r=records[key]
    audit.append({'i':key[0],'j':key[1],'visual_inliers':e['visual_inliers'],
                  'pnp_odom_diff_m':e['pnp_odom_diff_m'],
                  'icp_fitness':e['icp_fitness'],'reverse_fitness':e['reverse_fitness'],
                  'reaches_visual_anchor':key in anchors,'human_label':'','human_notes':''})
for page_idx in range(math.ceil(len(entries)/3)):
    page=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(page)
    for slot,e in enumerate(entries[page_idx*3:(page_idx+1)*3]):
        i,j=e['i'],e['j'];r=records[i,j];s=r['sp'];y=slot*row_h+10
        d.rectangle((8,y,W-8,y+row_h-7),fill='#fff0d6' if (i,j) in anchors else '#f3f3f3',outline='#888888',width=2)
        d.text((20,y+5),f'No-ICP extra {i}-{j} | visual {e["visual_inliers"]} | visual anchor: {"YES" if (i,j) in anchors else "no"}',fill='black',font=font)
        d.text((20,y+31),f'PnP/odom {e["pnp_odom_diff_m"]:.2f} m | LiDAR forward/reverse overlap {e["icp_fitness"]:.3f}/{e["reverse_fitness"]:.3f}',fill='black',font=small)
        cams=sorted(((int(c),int(n)) for c,n in s.get('per_cam',{}).items()),key=lambda x:(-x[1],x[0]))[:2]
        for k,(ci,n) in enumerate(cams):
            cj=(ci+int(r['shift']))%4
            for side,(node,cam,name) in enumerate(((i,ci,'query'),(j,cj,'match'))):
                x=20+(2*k+side)*tile_w
                path=frames/f'{node:04d}_c{cam}.png'
                if path.exists():
                    im=Image.open(path).convert('RGB');im.thumbnail((tile_w-10,tile_h-10))
                    page.paste(im,(x+(tile_w-10-im.width)//2,y+65+(tile_h-10-im.height)//2))
                d.text((x+4,y+370),f'{name} {node} cam{cam}'+(f' inliers {n}' if side==0 else ''),fill='black',font=small)
    page.save(root/f'review_page_{page_idx+1:02d}.jpg',quality=88)
with (root/'human_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
with zipfile.ZipFile(root/'review_pages.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.glob('review_page_*.jpg')):z.write(p,p.name)
    z.write(root/'human_audit.csv','human_audit.csv')
print('REVIEW',len(entries),'pages',math.ceil(len(entries)/3),'zip_bytes',(root/'review_pages.zip').stat().st_size)
