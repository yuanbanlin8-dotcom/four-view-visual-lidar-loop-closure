"""Complete and evaluate the NUC single-camera ablation after all workers finish."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929/full_single_camera_20261021')
python = '/home/nuc/.venvs/seeker-research312/bin/python'
deadline = time.monotonic() + 21600
while time.monotonic() < deadline:
    done = [(root / 'workers' / str(w) / 'complete.json').exists() for w in range(8)]
    if all(done):
        result = subprocess.run([python, str(root / 'scripts/run.py'), 'evaluate'],
                                cwd=root, check=False)
        (root / 'evaluation_exit_code.txt').write_text(str(result.returncode))
        raise SystemExit(result.returncode)
    for w, complete in enumerate(done):
        if complete:
            continue
        folder = root / 'workers' / str(w)
        pid = int((folder / 'pid').read_text())
        if not Path('/proc', str(pid)).exists():
            (root / 'watch_error.txt').write_text(f'worker {w} died, pid {pid}')
            raise SystemExit(2)
        state = (Path('/proc') / str(pid) / 'status').read_text().split('State:', 1)[1].strip()[:1]
        if state == 'T':
            os.kill(pid, 18)  # SIGCONT: the host intermittently suspends CPU workers.
            print('RESUMED', w, pid, flush=True)
    time.sleep(30)
(root / 'watch_error.txt').write_text('six-hour deadline exceeded')
raise SystemExit(3)
