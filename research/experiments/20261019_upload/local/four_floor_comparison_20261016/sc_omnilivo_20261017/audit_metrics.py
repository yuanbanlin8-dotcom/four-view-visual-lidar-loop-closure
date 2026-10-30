"""Audit local SC baseline artifacts using one pose-based metric throughout."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LABELS = json.loads((ROOT / 'floor_hypotheses.json').read_text())['labels']
ORDER = (2, 1, 0, 3)  # Physical floor order, low to high.
LASER_GAPS = np.array([4.18, 4.33, 4.18])


def pose_stats(filename):
    poses = np.loadtxt(ROOT / filename)
    assert poses.shape == (243, 8)
    medians = np.array([np.median(poses[np.array(LABELS, dtype=object) == level, 3])
                        for level in ORDER])
    spreads = np.array([
        np.percentile(poses[np.array(LABELS, dtype=object) == level, 3], 95)
        - np.percentile(poses[np.array(LABELS, dtype=object) == level, 3], 5)
        for level in ORDER
    ])
    gaps = np.diff(medians)
    return {'physical_floor_order': list(ORDER),
            'median_pose_z_m': medians.round(6).tolist(),
            'pose_z_p95_minus_p5_m': spreads.round(6).tolist(),
            'adjacent_median_pose_z_gaps_m': gaps.round(6).tolist(),
            'gap_mae_vs_laser_m': float(np.mean(np.abs(gaps - LASER_GAPS)))}


def main():
    retrieval = json.loads((ROOT / 'retrieval_summary.json').read_text())
    reports = [{k: v for k, v in item.items() if k != 'pairs'}
               for item in retrieval['reports']]
    geometry = [json.loads(line) for line in
                (ROOT / 'geometry_gap8_k3_thresh020.jsonl').read_text().splitlines()]
    assert len(geometry) == 204
    passing = [row for row in geometry if row['status'] == 'geometry_pass']
    assert len(passing) == 35
    relation = Counter(
        'unknown' if LABELS[row['i']] is None or LABELS[row['j']] is None
        else 'same' if LABELS[row['i']] == LABELS[row['j']] else 'cross'
        for row in passing
    )
    cross = [{'i': row['i'], 'j': row['j'],
              'floor_i': LABELS[row['i']], 'floor_j': LABELS[row['j']],
              'sc_distance': row['sc_distance'],
              'raw_position_distance_m': row['raw_position_distance_m'],
              'correction_translation_m': row['correction_translation_m'],
              'forward_fitness': row['forward_fitness'],
              'reverse_fitness': row['reverse_fitness']}
             for row in passing if LABELS[row['i']] is not None
             and LABELS[row['j']] is not None and LABELS[row['i']] != LABELS[row['j']]]
    raw = np.loadtxt(ROOT / 'raw_poses.txt')
    for row in cross:
        row['raw_z_difference_m'] = float(abs(raw[row['i'], 3] - raw[row['j'], 3]))
    output = {
        'retrieval_variants': reports,
        'geometry_candidate_count': len(geometry),
        'geometry_status_counts': dict(Counter(row['status'] for row in geometry)),
        'geometry_pass_floor_relation_posthoc': dict(relation),
        'known_cross_floor_geometry_pairs': cross,
        'pose_metric_note': 'Median keyframe pose Z per known floor; not mapped ground observations.',
        'laser_gap_readings_m': LASER_GAPS.tolist(),
        'pose_statistics': {
            'raw_omnilivo': pose_stats('raw_poses.txt'),
            'sc_plain_020': pose_stats('sc_plain_poses.txt'),
            'sc_cauchy_020': pose_stats('sc_cauchy_poses.txt'),
            'sc_plain_013': pose_stats('sc_strict_plain_poses.txt'),
            'sc_cauchy_013': pose_stats('sc_strict_cauchy_poses.txt'),
            'full_24s': pose_stats('full_gap24_poses.txt'),
        },
    }
    diagnostic_path = ROOT / 'suspect_edges_diagnostic_summary.json'
    if diagnostic_path.exists():
        cases = json.loads(diagnostic_path.read_text())['cases']
        output['posthoc_edge_deletion_diagnostics'] = {
            case: {'remaining_edge_count': data['remaining_edge_count'],
                   'variants': {name: pose_stats(f'sc_{case}_{name}_poses.txt')
                                for name in data['variants']}}
            for case, data in cases.items()
        }
    (ROOT / 'audit_summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items()
                      if k not in ('retrieval_variants', 'posthoc_edge_deletion_diagnostics')}, indent=2))
    if 'posthoc_edge_deletion_diagnostics' in output:
        print('Post-hoc diagnostic gap MAE (cm):')
        for case, data in output['posthoc_edge_deletion_diagnostics'].items():
            print(case, data['remaining_edge_count'],
                  {name: round(item['gap_mae_vs_laser_m'] * 100, 2)
                   for name, item in data['variants'].items()})


if __name__ == '__main__':
    main()
