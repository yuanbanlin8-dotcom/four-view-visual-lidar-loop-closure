"""Same-point-index outgoing/return ground-height audit on the 8 Oct flight.

Run on the NUC, where the two 153M-point unvoxelized PCDs reside. The
classroom ROI is fixed in the common local map frame; corridor logic matches
the previously archived full-corridor analysis. No per-pass Z recentering is
used for the numerical comparison.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/nuc/omnilivo_floor_loop_trial')
REPLAY = ROOT / 'recall_pose_v13_20261010/reintegrate_v14'
RAW = REPLAY / 'ws/src/Omni-LIVO/Log/PCD/all_raw_points.pcd'
FULL = ROOT / 'research_sync_20261012/fullres_colored/map_research_vision_pcm_gnc_floor_fullres_colored.pcd'
META = np.loadtxt(REPLAY / 'ws/src/Omni-LIVO/Log/PCD/scans_pos.json')
POSES = np.loadtxt(ROOT / 'independent_20261008_145008_recall_pose_v13/keyframe_poses.txt')
STARTS = np.r_[0, np.cumsum(META[:, 1].astype(np.int64))]
DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
OUT = Path(__file__).resolve().parent / 'z_audit.json'


def pcd(path):
    with path.open('rb') as f:
        while not f.readline().startswith(b'DATA '):
            pass
        offset = f.tell()
    return np.memmap(path, mode='r', dtype=DTYPE, offset=offset)


def pass_points(cloud, first, last):
    ids = np.flatnonzero((META[:, 0] >= POSES[first, 0] - 1.0)
                           & (META[:, 0] <= POSES[last, 0] + 1.0))
    return cloud[STARTS[ids[0]]:STARTS[ids[-1] + 1]]


def floor_cells(points, region):
    x, y, z = points['x'], points['y'], points['z']
    if region in ('classroom', 'classroom_center'):
        # Broad room ROI includes furniture and lower adjacent structures.
        # Center ROI isolates the common start/finish floor patch.
        if region == 'classroom_center':
            x0, x1, y0, y1 = -1.5, 1.5, -1.5, 1.5
            z0, z1 = -0.65, 0.0
        else:
            x0, x1, y0, y1 = -4., 4., -4., 4.
            z0, z1 = -3.0, 1.0
        grid = 0.5
        keep = ((x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                & (z > z0) & (z < z1))
        xx, yy, zz = x[keep], y[keep], z[keep]
        xi = np.floor((xx - x0)/grid).astype(np.int16)
        yi = np.floor((yy - y0)/grid).astype(np.int16)
        key = xi.astype(np.int32)*100 + yi
    else:
        # Identical full-length corridor strip and 0.5 m cells as Figure 3.
        x0, x1, y0, y1, grid = -1.7, 1.7, -26.5, -11., 0.5
        route = POSES[14:21]
        order = np.argsort(route[:, 2])
        center = np.interp(y, route[order, 2], route[order, 1])
        lateral = x - center
        keep = ((y >= y0) & (y < y1) & (np.abs(lateral) < 1.7)
                & (z > -1.7) & (z < 0.0))
        xx, yy, zz = lateral[keep], y[keep], z[keep]
        xi = np.floor((xx - x0)/grid).astype(np.int16)
        yi = np.floor((yy - y0)/grid).astype(np.int16)
        key = yi.astype(np.int32)*10 + xi
    del x, y, z
    cells = {}
    for k in np.unique(key):
        vals = zz[key == k]
        if len(vals) < 35:
            continue
        level = np.percentile(vals, 15)
        floor = vals[np.abs(vals - level) < 0.08]
        if len(floor) >= 15:
            cells[int(k)] = float(np.median(floor))
    return cells


def compare(first, second):
    keys = sorted(first.keys() & second.keys())
    return {k: second[k]-first[k] for k in keys}


def summaries(gaps):
    v = np.array(list(gaps.values()), dtype=float)
    return {'matched_cells': len(v), 'signed_median_m': float(np.median(v)),
            'median_abs_m': float(np.median(np.abs(v))),
            'p75_abs_m': float(np.percentile(np.abs(v), 75)),
            'p90_abs_m': float(np.percentile(np.abs(v), 90)),
            'fraction_under_5cm': float(np.mean(np.abs(v)<0.05))}


def main():
    clouds = {'raw': pcd(RAW), 'corrected': pcd(FULL)}
    assert len(clouds['raw']) == len(clouds['corrected']) == int(STARTS[-1])
    result = {'recording': '20261008_145008', 'raw_pcd': str(RAW),
              'corrected_pcd': str(FULL), 'point_count_each': int(STARTS[-1]),
              'regions': {}}
    for region, windows in [('classroom', [(0,7),(124,133)]),
                            ('classroom_center', [(0,7),(124,133)]),
                            ('corridor', [(14,20),(115,120)])]:
        cases = {}
        for name, cloud in clouds.items():
            a = floor_cells(pass_points(cloud, *windows[0]), region)
            b = floor_cells(pass_points(cloud, *windows[1]), region)
            cases[name] = compare(a,b)
        common = sorted(cases['raw'].keys() & cases['corrected'].keys())
        if len(common) < 10:
            raise RuntimeError(f'{region}: only {len(common)} common cells')
        # Both map cases use exactly the same cell IDs.
        common_cases = {name: {k: cases[name][k] for k in common}
                        for name in cases}
        row = {'windows': windows, 'cell_m': 0.5, 'identical_cells': True,
               'cases': {name: summaries(gaps) for name,gaps in common_cases.items()},
               'cells': [{'id': k, 'raw_gap_m': common_cases['raw'][k],
                          'corrected_gap_m': common_cases['corrected'][k]}
                         for k in common]}
        if region == 'corridor':
            for name in cases:
                stations = {}
                for k in common:
                    station = k//10
                    stations.setdefault(station, []).append(common_cases[name][k])
                row['cases'][name]['station_signed_median_m'] = [
                    {'y_m': -26.5+(i+0.5)*0.5,
                     'signed_gap_m': float(np.median(stations[i]))}
                    for i in sorted(stations)]
        result['regions'][region] = row
        print(region, json.dumps(row['cases']), flush=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('saved', OUT)


if __name__ == '__main__':
    main()
