"""Adapted LiDAR-only Scan Context pose graph on raw Omni-LIVO nodes.

Uses SC+ICP edges without project floor, visual, PCM, sequence or ground terms.
Runs plain and robust loop factors to expose sensitivity to false matches.
"""

import json
from pathlib import Path

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
OUT = ROOT / 'comparison_20261016/sc_raw_omnilivo_20261017'


def poses_as_matrices(rows):
    result = []
    for row in rows:
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        mat[:3, 3] = row[1:4]
        result.append(mat)
    return np.asarray(result)


def write_poses(path, times, poses):
    data = []
    for t, matrix in zip(times, poses):
        quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        data.append(np.r_[t, matrix[:3, 3], quat])
    np.savetxt(path, np.asarray(data), fmt='%.10f')


def floor_stats(matrices, labels):
    result = {}
    for level in sorted({x for x in labels if x is not None}):
        z = np.asarray([matrices[i, 2, 3] for i, x in enumerate(labels) if x == level])
        result[str(level)] = {'nodes': len(z), 'median_pose_z_m': float(np.median(z)),
                              'p95_p5_pose_z_m': float(np.percentile(z, 95) - np.percentile(z, 5))}
    return result


def solve(initial, edges, robust):
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    for i, pose in enumerate(initial):
        values.insert(i, gtsam.Pose3(pose))
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(initial[0]),
                                    gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.01] * 3 + [.05] * 3))
    loop_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([.03] * 3 + [.15] * 3))
    loop_noise = (gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), loop_base)
        if robust else loop_base)
    for i in range(1, len(initial)):
        relative = np.linalg.inv(initial[i - 1]) @ initial[i]
        graph.add(gtsam.BetweenFactorPose3(i - 1, i, gtsam.Pose3(relative), odom_noise))
    for edge in edges:
        graph.add(gtsam.BetweenFactorPose3(edge['j'], edge['i'],
                                          gtsam.Pose3(np.asarray(edge['relative_transform'])),
                                          loop_noise))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    final_values = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    return np.asarray([final_values.atPose3(i).matrix() for i in range(len(initial))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = np.loadtxt(ROOT / 'loop_latest_20261014/raw_poses.txt')
    initial = poses_as_matrices(raw)
    edges = [json.loads(line) for line in (OUT / 'geometry_gap8_k3_thresh020.jsonl').read_text().splitlines()]
    edges = [x for x in edges if x['status'] == 'geometry_pass']
    labels = json.loads((ROOT / 'loop_latest_20261014/floor_hypotheses.json').read_text())['labels']
    report = {'input': '243 raw Omni-LIVO nodes; SC gap8 K3 threshold0.20 + bidirectional ICP',
              'edge_count': len(edges), 'graph': 'raw odometry + SC ICP loops, no floor/visual/PCM/sequence/ground factors',
              'raw_floor_pose_statistics': floor_stats(initial, labels), 'variants': {}}
    for threshold, prefix in ((.20, ''), (.13, 'strict_')):
        chosen = [x for x in edges if x['sc_distance'] < threshold]
        for robust in (False, True):
            name = prefix + ('plain' if not robust else 'cauchy')
            poses = solve(initial, chosen, robust)
            write_poses(OUT / f'sc_{name}_poses.txt', raw[:, 0], poses)
            shift = np.linalg.norm(poses[:, :3, 3] - initial[:, :3, 3], axis=1)
            dz = np.abs(poses[:, 2, 3] - initial[:, 2, 3])
            report['variants'][name] = {
                'edge_count': len(chosen),
                'sc_threshold': threshold,
                'median_node_shift_m': float(np.median(shift)),
                'max_node_shift_m': float(np.max(shift)),
                'max_vertical_shift_m': float(np.max(dz)),
                'floor_pose_statistics': floor_stats(poses, labels),
            }
            print(name, report['variants'][name], flush=True)
    (OUT / 'pose_graph_summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
