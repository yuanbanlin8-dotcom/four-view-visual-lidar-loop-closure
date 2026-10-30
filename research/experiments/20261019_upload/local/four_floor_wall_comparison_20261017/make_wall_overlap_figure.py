"""Upper-storey same-wall pass separation: original Omni-LIVO vs full pipeline.

Run on the NUC. The two PCDs use the same recording but different frontends,
coordinates, point counts, and reconstruction. This is a whole-system example.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929")
OUT = Path(__file__).resolve().parent
DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
CASES = [
    {
        "name": "Original Omni-LIVO",
        "pcd": ROOT / "frontend_ablation_20261014/legacy_livo_full_0.5x/map_raw.pcd",
        "y0": 237.0,
        "y1": 238.5,
        "x_ref": lambda y: 63.6 + 0.09 * (y - 232.0),
        "z_ref": lambda y: -94.0 - 0.43 * (y - 232.0),
        "slope": 0.09,
        "passes": [(190_000_000, 215_000_000), (225_000_000, 245_000_000)],
    },
    {
        "name": "Complete 24 s method",
        "pcd": ROOT / "sequence_gap_ablation_20261014/full_resolution/map_four_floors_20261013_loop24s_corrected_fullres_colored.pcd",
        "y0": 19.0,
        "y1": 20.5,
        "x_ref": lambda y: 10.3 + 0.20 * (y - 14.0),
        "z_ref": None,
        "slope": 0.20,
        "passes": [(215_000_000, 245_000_000), (250_000_000, 280_000_000)],
    },
]


def map_pcd(path):
    with path.open("rb") as f:
        while not f.readline().startswith(b"DATA "):
            pass
        offset = f.tell()
    return np.memmap(path, dtype=DTYPE, mode="r", offset=offset)


def select_pass(cloud, case, interval):
    ys, normals = [], []
    lo, hi = interval
    for start in range(lo, min(hi, len(cloud)), 2_000_000):
        piece = cloud[start : min(start + 2_000_000, hi, len(cloud))]
        y = piece["y"]
        basic = (y >= case["y0"]) & (y < case["y1"])
        if not basic.any():
            continue
        y = y[basic]
        x = piece["x"][basic]
        z = piece["z"][basic]
        # Orthogonal distance to the manually identified local wall line.
        normal = (x - case["x_ref"](y)) / np.sqrt(1 + case["slope"] ** 2)
        good = (normal > -0.30) & (normal < 0.30)
        if case["z_ref"] is None:
            good &= (z > 4.0) & (z < 6.2)
        else:
            zr = z - case["z_ref"](y)
            good &= (zr > -1.3) & (zr < 0.7)
        ys.append(y[good].copy())
        normals.append(normal[good].copy())
    return np.concatenate(ys), np.concatenate(normals)


def station_profile(case, first, second):
    stations, a, b, counts = [], [], [], []
    for y0 in np.arange(case["y0"], case["y1"], 0.10):
        aa = first[1][(first[0] >= y0) & (first[0] < y0 + 0.10)]
        bb = second[1][(second[0] >= y0) & (second[0] < y0 + 0.10)]
        if min(len(aa), len(bb)) >= 100:
            stations.append(y0 - case["y0"] + 0.05)
            a.append(np.median(aa))
            b.append(np.median(bb))
            counts.append([len(aa), len(bb)])
    if len(stations) < 8:
        raise RuntimeError(f"Insufficient shared wall stations for {case['name']}")
    return np.array(stations), np.array(a), np.array(b), counts


def main():
    results = []
    for case in CASES:
        cloud = map_pcd(case["pcd"])
        first, second = [select_pass(cloud, case, window) for window in case["passes"]]
        stations, a, b, counts = station_profile(case, first, second)
        gaps = b - a
        results.append((case, first, second, stations, a, b, gaps, counts))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=200, sharex=True, sharey=True)
    for ax, (case, first, second, stations, a, b, gaps, counts) in zip(axes, results):
        baseline = np.median(a)
        ax.plot(stations, (a - baseline) * 100, color="#1870a5", lw=2.2,
                marker="o", ms=3, label="First pass")
        ax.plot(stations, (b - baseline) * 100, color="#df8f1e", lw=2.2,
                marker="o", ms=3, label="Revisit")
        ax.fill_between(stations, (a - baseline) * 100,
                        (b - baseline) * 100,
                        color="0.55", alpha=0.17)
        ax.set_title(case["name"], fontweight="bold")
        ax.set_xlabel("Distance along selected wall section (m)")
        ax.grid(alpha=0.2)
        ax.text(0.03, 0.04,
                f"{len(stations)} paired 0.10 m stations\n"
                f"Median wall-pass gap: {np.median(np.abs(gaps))*100:.1f} cm",
                transform=ax.transAxes, va="bottom", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92))
    axes[0].set_ylabel("Wall-normal offset from first-pass median (cm)")
    axes[1].legend(loc="upper right")
    axes[0].set_xlim(0, 1.5)
    fig.suptitle("Four-storey map: repeated upper-storey corridor wall",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    stem = OUT / "upper_storey_wall_overlap_before_after"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    report = {
        "recording": "20261013_160929",
        "site": "F4 left-side wall beside accepted revisit 191-168",
        "metric": "Median of absolute differences between pass-wise wall-normal medians at common 0.10 m stations",
        "comparison_scope": "Whole-system original Omni-LIVO replay versus complete frozen 24 s map; different frontend configurations and reconstruction",
        "pass_assignment": "Earlier/later chronological PCD point-index clusters, cross-checked against trajectory revisit 168/191; original replay has no explicit per-point timestamps",
        "cases": [
            {
                "name": c["name"],
                "pcd": str(c["pcd"]),
                "y_range_m": [c["y0"], c["y1"]],
                "point_index_windows": c["passes"],
                "first_points": len(first[0]),
                "revisit_points": len(second[0]),
                "paired_stations": len(stations),
                "median_abs_gap_m": float(np.median(np.abs(gaps))),
                "p75_abs_gap_m": float(np.percentile(np.abs(gaps), 75)),
                "max_abs_station_gap_m": float(np.max(np.abs(gaps))),
                "normal_distance_reference": "Local wall centreline x(y); common reference within each PCD",
            }
            for c, first, second, stations, _, _, gaps, _ in results
        ],
        "limitation": "Selected illustrative wall patch. Not physical wall thickness or surveyed 3D accuracy. A different map-coordinate crop is required for each PCD due to global drift.",
    }
    (OUT / "upper_storey_wall_overlap_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
