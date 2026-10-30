from pathlib import Path
import json
import numpy as np

root = Path('/home/nuc/omnilivo_floor_loop_trial')
replay = root / 'recall_pose_v13_20261010/reintegrate_v14/ws/src/Omni-LIVO/Log/PCD'
raw = replay / 'all_raw_points.pcd'
full = root / 'research_sync_20261012/fullres_colored/map_research_vision_pcm_gnc_floor_fullres_colored.pcd'
poses = np.loadtxt(root / 'independent_20261008_145008_recall_pose_v13/keyframe_poses.txt')
meta = np.loadtxt(replay / 'scans_pos.json')
for name, path in [('raw', raw), ('corrected', full)]:
    print(name, path.exists(), path.stat().st_size if path.exists() else None)
    if path.exists():
        with path.open('rb') as f:
            header = []
            while True:
                line = f.readline()
                header.append(line.decode(errors='replace').strip())
                if line.startswith(b'DATA '): break
        print('header', header)
print('pose shape', poses.shape, 'meta shape', meta.shape)
print('pose min/max x y z', poses[:,1:4].min(axis=0), poses[:,1:4].max(axis=0))
for i in list(range(0,25,2)) + list(range(100,134,3)):
    if i < len(poses):
        print(i, poses[i,0], *(round(v,3) for v in poses[i,1:4]))
