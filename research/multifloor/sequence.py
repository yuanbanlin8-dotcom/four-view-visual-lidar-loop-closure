"""Complete-link, same-level sequence support without borrowing duplicate frames."""
import numpy as np
from scipy.spatial.transform import Rotation


def correction(row, poses):
    i, j = row['query'], row['match']
    return poses[j] @ np.asarray(row['sp']['icp_T']) @ np.linalg.inv(poses[i])


def consistent(a, b, translation=0.8, angle_deg=8.):
    d = np.linalg.inv(a) @ b
    return (np.linalg.norm(d[:3,3]) <= translation
            and Rotation.from_matrix(d[:3,:3]).magnitude() <= np.radians(angle_deg))


def supported_sequences(rows, poses, times, labels, min_pairs=3, min_travel=6., max_gap_sec=24.):
    """Every row must already pass PnP and ICP. Sequences cannot rescue failed PnP."""
    paths = []
    corrections = [correction(r, poses) for r in rows]
    order = sorted(range(len(rows)), key=lambda k: (rows[k]['query'], rows[k]['match']))
    for direction in (-1, 1):
        active = []
        for k in order:
            r = rows[k]
            i, j = r['query'], r['match']
            if labels[i] is None or labels[i] != labels[j]:
                continue
            compatible = []
            for path in active:
                last = rows[path[-1]]
                q, m = last['query'], last['match']
                if not (0 < times[i]-times[q] <= max_gap_sec
                        and 0 < direction*(times[j]-times[m]) <= max_gap_sec*1.5):
                    continue
                if labels[q] != labels[i]:
                    continue
                # Complete-link: A~B and B~C cannot certify A~C.
                if all(consistent(corrections[k], corrections[p]) for p in path):
                    compatible.append(path)
            if compatible:
                path = max(compatible, key=lambda p: (len(p), sum(rows[x]['sp']['inliers'] for x in p))) + [k]
            else:
                path = [k]
            active.append(path)
            if len(path) < min_pairs:
                continue
            source = np.array([poses[rows[p]['query']][:3,3] for p in path])
            target = np.array([poses[rows[p]['match']][:3,3] for p in path])
            # Spatial span, not accumulated jitter during a stationary pause.
            span = min(np.linalg.norm(source[-1]-source[0]), np.linalg.norm(target[-1]-target[0]))
            if span >= min_travel:
                paths.append(dict(direction=direction, members=path, span_m=float(span)))
    supported = {}
    for cid, path in enumerate(sorted(paths, key=lambda p: (-len(p['members']), p['members']))):
        for k in path['members']:
            if k not in supported:
                supported[k] = dict(sequence_id=cid, sequence_pairs=len(path['members']),
                                    sequence_direction=path['direction'], sequence_span_m=path['span_m'])
    return [dict(row, **supported[k]) for k, row in enumerate(rows) if k in supported]
