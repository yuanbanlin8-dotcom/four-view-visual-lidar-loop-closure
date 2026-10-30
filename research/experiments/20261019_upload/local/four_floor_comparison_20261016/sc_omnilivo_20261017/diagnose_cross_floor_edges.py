"""Post-hoc graph diagnostics for suspect SC constraints.

These deliberately use floor labels after SC retrieval and are not SC baselines.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sc_raw_omnilivo_20261017_pose_graph import (ROOT, OUT, floor_stats,
                                                 poses_as_matrices, solve, write_poses)


BAD_PAIRS = {(92, 43), (122, 68)}
LARGE_PAIR = {(152, 143)}


def main():
    raw = np.loadtxt(ROOT / 'loop_latest_20261014/raw_poses.txt')
    initial = poses_as_matrices(raw)
    edges = [json.loads(line) for line in
             (OUT / 'geometry_gap8_k3_thresh020.jsonl').read_text().splitlines()]
    edges = [x for x in edges if x['status'] == 'geometry_pass']
    assert len(edges) == 35
    labels = json.loads((ROOT / 'loop_latest_20261014/floor_hypotheses.json').read_text())['labels']
    cases = {
        'drop_two_cross': [x for x in edges if (x['i'], x['j']) not in BAD_PAIRS],
        'drop_large_nearby': [x for x in edges if (x['i'], x['j']) not in LARGE_PAIR],
        'drop_three_suspect': [x for x in edges if (x['i'], x['j']) not in BAD_PAIRS | LARGE_PAIR],
        'known_same_floor_only': [x for x in edges if labels[x['i']] is not None
                                  and labels[x['i']] == labels[x['j']]],
    }
    output = {'diagnostic_only': True, 'cases': {}}
    for case, chosen in cases.items():
        output['cases'][case] = {'remaining_edge_count': len(chosen), 'variants': {}}
        for robust in (False, True):
            name = 'cauchy' if robust else 'plain'
            poses = solve(initial, chosen, robust)
            write_poses(OUT / f'sc_{case}_{name}_poses.txt', raw[:, 0], poses)
            output['cases'][case]['variants'][name] = floor_stats(poses, labels)
    (OUT / 'suspect_edges_diagnostic_summary.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
