"""Diagnostic overlap of raw-pose-transformed SC candidate submaps."""

from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
POSES = np.loadtxt(ROOT / "raw_poses.txt")
SUBMAPS = ROOT.parent.parent / "four_floor_gpu" / "dataset" / "submaps"


def points(index):
    local = np.load(SUBMAPS / f"{index:04d}.npy").astype(np.float64)
    pose = POSES[index]
    return Rotation.from_quat(pose[4:8]).apply(local) + pose[1:4]


def inspect(i, j):
    a, b = points(i), points(j)
    distance, nearest = cKDTree(b).query(a, distance_upper_bound=0.3)
    matched = np.isfinite(distance)
    heights = a[matched, 2]
    bins = np.arange(-12, 9, 1)
    hist, _ = np.histogram(heights, bins=bins)
    top = sorted(zip(hist, bins[:-1]), reverse=True)[:6]
    print(i, j, "overlap", matched.sum(), len(a), "fraction", round(matched.mean(), 3))
    print("  matched global z bins", [(int(z), int(n)) for n, z in top])
    print("  matched global z percentiles", np.round(np.percentile(heights, [5, 25, 50, 75, 95]), 2))


for pair in ((92, 43), (122, 68), (152, 143), (77, 37), (140, 79), (232, 2)):
    inspect(*pair)


print("\nGeometry corrections expressed in the raw world frame:")
rows = [json.loads(line) for line in (ROOT / "geometry_gap8_k3_thresh020.jsonl").read_text().splitlines()]
rows = [row for row in rows if row["status"] == "geometry_pass"]
for row in sorted(rows, key=lambda r: -r["correction_translation_m"]):
    i, j = row["i"], row["j"]
    refined = np.array(row["relative_transform"])
    world_j = np.eye(4)
    world_j[:3, :3] = Rotation.from_quat(POSES[j, 4:8]).as_matrix()
    world_j[:3, 3] = POSES[j, 1:4]
    world_i_refined = world_j @ refined
    world_delta = world_i_refined[:3, 3] - POSES[i, 1:4]
    if row["correction_translation_m"] > 0.1:
        print(i, j, "raw_dz", round(POSES[i, 3] - POSES[j, 3], 2),
              "world_delta_xyz", np.round(world_delta, 2),
              "fitness", round(row["forward_fitness"], 2),
              round(row["reverse_fitness"], 2))


print("\nRaw versus ICP-refined point overlap for highest-risk pairs:")
for i, j in ((152, 143), (92, 43), (122, 68), (210, 31)):
    row = next(row for row in rows if (row["i"], row["j"]) == (i, j))
    local_i = np.load(SUBMAPS / f"{i:04d}.npy").astype(np.float64)
    local_j = np.load(SUBMAPS / f"{j:04d}.npy").astype(np.float64)
    raw_i = points(i)
    raw_j = points(j)
    tree = cKDTree(raw_j)
    raw_near = np.isfinite(tree.query(raw_i, distance_upper_bound=0.3)[0]).mean()
    refined = np.array(row["relative_transform"])
    world_j = np.eye(4)
    world_j[:3, :3] = Rotation.from_quat(POSES[j, 4:8]).as_matrix()
    world_j[:3, 3] = POSES[j, 1:4]
    world_i = world_j @ refined
    aligned_i = local_i @ world_i[:3, :3].T + world_i[:3, 3]
    refined_near = np.isfinite(tree.query(aligned_i, distance_upper_bound=0.3)[0]).mean()
    print(i, j, round(raw_near, 3), round(refined_near, 3))
