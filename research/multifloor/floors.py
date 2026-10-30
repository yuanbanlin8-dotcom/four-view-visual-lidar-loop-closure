"""Conservative temporal floor hypotheses from measured ground planes.

No fixed storey height, global Z bins, or query/database lap numbers. A landing
only becomes a level after sustained horizontal travel. Missing or ambiguous
evidence stays unknown. This is an offline classifier using a centred window.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FloorSettings:
    window_seconds: float = 9.0
    minimum_duration: float = 12.0
    minimum_travel: float = 6.0
    minimum_samples: int = 5
    maximum_local_range: float = 0.60
    maximum_slope_mps: float = 0.07
    maximum_tilt_deg: float = 12.0
    maximum_ground_rmse: float = 0.06
    same_level_tolerance: float = 0.9
    minimum_level_separation: float = 1.8


def classify_floors(times, positions, measurements, settings=FloorSettings()):
    times = np.asarray(times, float)
    positions = np.asarray(positions, float)
    n = len(times)
    if positions.shape != (n, 3) or len(measurements) != n:
        raise ValueError('Ground measurements must correspond to every pose')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Pose timestamps must be finite and strictly increasing')
    z = np.full(n, np.nan)
    reliable = np.zeros(n, bool)
    for i, m in enumerate(measurements):
        if not m:
            continue
        if (m.get('inliers', 0) >= 150 and m.get('tilt_deg', 90) <= settings.maximum_tilt_deg
                and m.get('rmse', float('inf')) <= settings.maximum_ground_rmse
                and np.isfinite(m.get('z_floor', np.nan))):
            z[i] = m['z_floor']
            reliable[i] = True
    stable = np.zeros(n, bool)
    for i in range(n):
        ids = np.flatnonzero((abs(times-times[i]) <= settings.window_seconds) & reliable)
        if not reliable[i] or len(ids) < 3:
            continue
        values = z[ids]
        slope = np.polyfit(times[ids]-times[i], values, 1)[0]
        stable[i] = (np.ptp(values) <= settings.maximum_local_range
                     and abs(slope) <= settings.maximum_slope_mps)
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions[:,:2], axis=0), axis=1))]
    # A missing sample never supplies a floor identity; short holes may occur
    # within a plateau, but the segment cannot bridge a long transition.
    groups = []
    for i in np.flatnonzero(stable):
        if groups and i-groups[-1][-1] <= 2 and times[i]-times[groups[-1][-1]] <= 9:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    labels = [None] * n
    levels, segments, transitions = [], [], []
    for group in groups:
        a, b = group[0], group[-1]
        if (len(group) < settings.minimum_samples or times[b]-times[a] < settings.minimum_duration
                or travel[b]-travel[a] < settings.minimum_travel):
            continue
        center = float(np.median(z[group]))
        spread = float(np.percentile(z[group], 90)-np.percentile(z[group], 10))
        compatible = [k for k, h in enumerate(levels) if abs(h-center) <= settings.same_level_tolerance]
        if len(compatible) == 1:
            level = compatible[0]
        elif not compatible and all(abs(h-center) >= settings.minimum_level_separation for h in levels):
            level = len(levels)
            levels.append(center)
        else:
            # Do not silently merge a half landing with an occupied floor.
            continue
        for i in group:
            labels[i] = level
        segment = dict(id=len(segments), level=level, start=a, end=b, nodes=group,
                       ground_center_m=center, ground_spread_m=spread,
                       travel_m=float(travel[b]-travel[a]))
        if segments:
            prev = segments[-1]
            if prev['level'] != level:
                transitions.append(dict(start=prev['end'], end=a,
                                        from_level=prev['level'], to_level=level,
                                        delta_m=center-prev['ground_center_m'],
                                        sigma_m=max(.3, (spread+prev['ground_spread_m'])/2),
                                        kind='observed_plateau_connection'))
        segments.append(segment)
    return dict(labels=labels, levels=levels, segments=segments, transitions=transitions,
                unknown_nodes=[i for i, v in enumerate(labels) if v is None],
                mode='offline_temporal_ground_plateaus', ground_z=z.tolist(),
                settings=vars(settings),
                limitation='Level labels depend on raw odometry and ground fits; uncertain nodes cannot certify same-level loops.')


def floor_relation(i, j, labels):
    a, b = labels[i], labels[j]
    if a is None or b is None:
        return 'unknown'
    return 'same' if a == b else 'different'
