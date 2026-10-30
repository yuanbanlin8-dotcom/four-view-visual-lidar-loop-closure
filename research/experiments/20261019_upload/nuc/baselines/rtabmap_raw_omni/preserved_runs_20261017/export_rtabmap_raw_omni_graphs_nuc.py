from pathlib import Path
import os,subprocess,csv
base=Path('/home/nuc/omnilivo_floor_loop_trial/external_baselines_20261017')
r=base/'four_floor_rtabmap_raw_omni';exe=base/'four_floor_rtabmap_multicam/build/export_rtabmap_graph'
env=os.environ.copy();env['LD_LIBRARY_PATH']='/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH','')
for mode in (0,1,2,3,4):
 stem='rtabmap_multicam_raw_omni'+(f'_mode{mode}' if mode else '')
 p=subprocess.run([str(exe),str(r/f'{stem}.db'),str(r/f'{stem}_path')],capture_output=True,text=True,env=env,timeout=120)
 print('EXPORT',mode,p.returncode,p.stderr[-350:],flush=True)
 if p.returncode:continue
 for typ in ('raw','optimized'):
  with (r/f'{stem}_path_{typ}.csv').open() as f:rows=list(csv.DictReader(f))
  print('PATH',mode,typ,len(rows),'return_z',float(rows[-1]['z'])-float(rows[0]['z']),flush=True)
