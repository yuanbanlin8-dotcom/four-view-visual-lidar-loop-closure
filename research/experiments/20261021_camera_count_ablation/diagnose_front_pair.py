"""Audit per-camera PnP depth support in the completed c0+c1 ablation."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
import run


def main():
    work = run.WORK
    for w in range(8):
        folder = work / 'workers' / str(w)
        run.setup(folder / 'vision')
        import pnp_multicam as pn
        records = json.loads((folder / 'candidates.json').read_text())
        for r in records:
            if r['sp'].get('inliers', 0) < 15:
                continue
            i, j = r['query'], r['match']
            parts = []
            for f in sorted((folder / 'vision/data/features').glob(
                    f'frontpair_c01_{i}_{j}_c*_sp.npz')):
                cj = int(f.name.split('_c')[1][1])
                with np.load(f) as z:
                    bearings = z['bb']
                    lifted = pn.lift(bearings, j, cj)
                    parts.append(dict(camera_pair=f.name.split('_c')[1][:2],
                                      essential_matches=len(bearings),
                                      lifted=int(np.isfinite(lifted).all(axis=1).sum())))
            print(json.dumps(dict(query=i, match=j,
                                  pnp_inliers=r['sp']['inliers'],
                                  pnp_per_cam=r['sp'].get('per_cam'),
                                  parts=parts)))


if __name__ == '__main__':
    main()
