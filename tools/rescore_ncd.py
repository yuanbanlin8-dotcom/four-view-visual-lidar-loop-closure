"""Re-score included NCD outputs without raw sensor data."""
from pathlib import Path
import subprocess, sys, json
root=Path(__file__).resolve().parents[1]
data=root/'research/eval/ncd_quad_easy/sot_samefloor100'
out=root/'outputs/ncd_rescore'
subprocess.run([sys.executable,str(root/'research/eval/scripts/ncd_quad_easy_metrics.py'),
    '--gt-poses',str(data/'keyframe_poses.txt'),'--vision-dir',str(data),
    '--backend-dir',str(data),'--out',str(out)],cwd=root,check=True)
metrics=json.loads((out/'metrics_ncd_quad_easy_adapter3s.json').read_text())
assert metrics['R_at_k']['hits@1']==30
assert metrics['R_at_k']['hits@5']==31
assert metrics['n_accepted_edges']==100
assert metrics['loop_pr']['tp']==96
assert metrics['loop_pr']['covered_pos']==32
assert metrics['loop_pr']['covered_reverse']==0
print('NCD score check passed.')
