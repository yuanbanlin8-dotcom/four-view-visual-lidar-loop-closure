from pathlib import Path
import subprocess,time,traceback

root=Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929/comparison_20261016/four_independent')
python='/home/nuc/.venvs/seeker-research312/bin/python'
deadline=time.monotonic()+10800
while time.monotonic()<deadline:
    if all((root/'workers'/str(k)/'complete.json').exists() for k in range(4)):
        print('ALL_WORKERS_COMPLETE',flush=True)
        result=subprocess.run([python,str(root/'evaluate.py')],check=False)
        print('EVALUATION_EXIT',result.returncode,flush=True)
        raise SystemExit(result.returncode)
    for k in range(4):
        work=root/'workers'/str(k)
        if not (work/'complete.json').exists():
            pid=int((work/'pid').read_text())
            if not Path('/proc',str(pid)).exists():
                print('WORKER_DIED',k,pid,flush=True)
                raise SystemExit(2)
    time.sleep(20)
print('WATCH_TIMEOUT',flush=True)
raise SystemExit(3)
