"""Compare October 8 corridor ground in same-point raw and full-method PCDs.

Run on the NUC; both PCDs have identical point order and no voxel reduction.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/nuc/omnilivo_floor_loop_trial")
REPLAY = ROOT / "recall_pose_v13_20261010/reintegrate_v14"
BASE = ROOT / "independent_20261008_145008_recall_pose_v13"
FULL = ROOT / "research_sync_20261012/fullres_colored"
OUT = Path(__file__).resolve().parent
RAW_PCD = REPLAY / "ws/src/Omni-LIVO/Log/PCD/all_raw_points.pcd"
FULL_PCD = FULL / "map_research_vision_pcm_gnc_floor_fullres_colored.pcd"
META = np.loadtxt(REPLAY / "ws/src/Omni-LIVO/Log/PCD/scans_pos.json")
POSES = np.loadtxt(BASE / "keyframe_poses.txt")
STARTS = np.r_[0, np.cumsum(META[:, 1].astype(np.int64))]
DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
Y_MIN, Y_MAX = -26.5, -11.0
GRID = 0.5


def pcd(path):
    with path.open("rb") as f:
        while not f.readline().startswith(b"DATA "):
            pass
        offset = f.tell()
    return np.memmap(path, mode="r", dtype=DTYPE, offset=offset)


def pass_points(cloud, first, last):
    ids = np.flatnonzero(
        (META[:, 0] >= POSES[first, 0] - 1.0)
        & (META[:, 0] <= POSES[last, 0] + 1.0)
    )
    return cloud[STARTS[ids[0]] : STARTS[ids[-1] + 1]]


def ground_cells(points):
    # The outgoing pass defines the physical corridor centreline. A generous
    # strip accommodates the small horizontal correction on the return pass.
    route = POSES[14:21]
    order = np.argsort(route[:, 2])
    y = points["y"]
    z = points["z"]
    x_center = np.interp(y, route[order, 2], route[order, 1])
    lateral = points["x"] - x_center
    keep = (
        (y >= Y_MIN)
        & (y < Y_MAX)
        & (np.abs(lateral) < 1.7)
        & (z > -1.7)
        & (z < 0.0)
    )
    y, lateral, z = y[keep], lateral[keep], z[keep]
    yi = ((y - Y_MIN) / GRID).astype(np.int16)
    xi = ((lateral + 1.7) / GRID).astype(np.int8)
    cell = yi.astype(np.int32) * 10 + xi
    rows = []
    for key in np.unique(cell):
        zz = z[cell == key]
        if len(zz) < 35:
            continue
        lo = float(np.percentile(zz, 15))
        floor = zz[np.abs(zz - lo) < 0.08]
        if len(floor) < 15:
            continue
        rows.append((int(key // 10), int(key % 10), float(np.median(floor)), len(floor)))
    return np.array(rows, dtype=float).reshape(-1, 4)


def compare(first, second):
    a = {(int(r[0]), int(r[1])): r[2] for r in first}
    b = {(int(r[0]), int(r[1])): r[2] for r in second}
    common = sorted(set(a) & set(b))
    gaps = np.array([b[k] - a[k] for k in common])
    return common, gaps


def station_curve(cells):
    stations = np.unique(cells[:, 0].astype(int))
    return (
        Y_MIN + (stations + 0.5) * GRID,
        np.array([np.median(cells[cells[:, 0] == i, 2]) for i in stations]),
    )


def main():
    raw = pcd(RAW_PCD)
    corrected = pcd(FULL_PCD)
    assert len(raw) == len(corrected) == int(STARTS[-1])
    cases = []
    for name, cloud in [("Omni-LIVO, before loop closure", raw),
                        ("Full method, corrected", corrected)]:
        outward = ground_cells(pass_points(cloud, 14, 20))
        returning = ground_cells(pass_points(cloud, 115, 120))
        common, gaps = compare(outward, returning)
        if len(gaps) < 20:
            raise RuntimeError(f"Too few shared ground cells: {name}: {len(gaps)}")
        cases.append((name, outward, returning, common, gaps))

    # Use the identical spatial cells for both before/after gap statistics.
    same_cells = sorted(set(cases[0][3]) & set(cases[1][3]))
    comparable = []
    for name, outward, returning, common, gaps in cases:
        gap_by_cell = dict(zip(common, gaps))
        comparable.append((name, outward, returning, same_cells,
                           np.array([gap_by_cell[k] for k in same_cells])))
    cases = comparable

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=200, sharey=True)
    colors = ("#1870a5", "#df8f1e")
    for ax, (name, outward, returning, common, gaps) in zip(axes, cases):
        baseline = np.median(outward[:, 2])
        curves = []
        for cells, label, color in [(outward, "Outward pass", colors[0]),
                                    (returning, "Return pass", colors[1])]:
            xx, zz = station_curve(cells)
            ax.plot(xx, zz - baseline, color=color, lw=2.2, label=label)
            curves.append((xx, zz - baseline))
        # The shaded area makes the two mapped ground surfaces easy to locate.
        shared_y = np.intersect1d(curves[0][0], curves[1][0])
        ax.fill_between(shared_y,
                        np.interp(shared_y, curves[0][0], curves[0][1]),
                        np.interp(shared_y, curves[1][0], curves[1][1]),
                        color="0.65", alpha=0.16)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Position along long corridor, map Y (m)")
        ax.set_xlim(Y_MIN, Y_MAX)
        ax.grid(alpha=0.2)
        ax.text(0.03, 0.04,
                f"Matched 0.5 m ground cells: {len(gaps)}\n"
                f"Median |return - outward|: {np.median(np.abs(gaps))*100:.1f} cm",
                transform=ax.transAxes, va="bottom", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9))
    axes[0].set_ylabel("Ground height relative to outward-pass median (m)")
    axes[1].legend(loc="upper right")
    fig.suptitle("8 October single-storey route: repeated ground in the long corridor",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    png = OUT / "single_floor_long_corridor_ground_before_after.png"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(OUT / "single_floor_long_corridor_ground_before_after.pdf",
                bbox_inches="tight")
    plt.close(fig)
    report = {
        "recording": "20261008_145008",
        "before_pcd": str(RAW_PCD),
        "after_pcd": str(FULL_PCD),
        "point_count_each": int(len(raw)),
        "frame_indices_outward": [14, 20],
        "frame_indices_return": [115, 120],
        "corridor_map_y_m": [Y_MIN, Y_MAX],
        "ground_cell_size_m": GRID,
        "identical_spatial_cells_for_both_cases": True,
        "cases": [
            {"name": name, "matched_cells": len(common),
             "signed_median_gap_m": float(np.median(gaps)),
             "median_abs_gap_m": float(np.median(np.abs(gaps))),
             "p75_abs_gap_m": float(np.percentile(np.abs(gaps), 75))}
            for name, _, _, common, gaps in cases
        ],
        "interpretation": "Same-recording, same-point-order whole-pipeline map consistency; not surveyed floor accuracy.",
    }
    (OUT / "single_floor_long_corridor_ground_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["cases"], indent=2))
    print(png)


if __name__ == "__main__":
    main()
