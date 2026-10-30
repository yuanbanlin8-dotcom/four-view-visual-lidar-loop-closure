"""Reproduce key counts for the corrected c0+c1 flight ablation."""
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / 'corrected_front_pair'
REFERENCE = HERE.parent / '20261019_upload/nuc/four_floor/sequence_gap/gap_24_summary.json'


def main():
    records = sum((json.loads(path.read_text()) for path in sorted(
        RAW.glob('candidates_worker_*.json'))), [])
    shortlist = json.loads((RAW / 'shortlist.json').read_text())
    summary = json.loads((RAW / 'camera_c4_summary.json').read_text())
    decisions = json.loads((RAW / 'camera_c4_gate_audit.json').read_text())
    reference = json.loads(REFERENCE.read_text())
    ref_edges = {(e['i'], e['j']) for e in reference['accepted_edges']
                 if e['gnc_weight'] > 0.5}
    key = lambda row: (row['query'], row['match'])
    assert len(records) == len(shortlist) == len(decisions) == 1100
    assert {key(r) for r in records} == {key(r) for r in shortlist}
    assert {key(r) for r in records} == {key(r) for r in decisions}
    assert all(r['camera'] == 4 for r in records)
    assert sum(d['decision'] == 'pair_pass' for d in decisions) == summary['pair_pass']
    assert summary['accepted_loops'] == len(summary['accepted_edges']) == 0

    visual_supported = [r for r in records
                        if r['sp'].get('inliers', 0) >= 15
                        and sum(v >= 5 for v in r['sp'].get('per_cam', {}).values()) >= 2]
    assert len(visual_supported) == 5
    retrieved = {key(r) for r in records} & ref_edges
    result = dict(
        flight='four_floor_20261013_160929', cameras=['c0', 'c1'],
        shortlisted_candidates=len(records),
        reference_four_view_accepted=len(ref_edges),
        reference_edges_retrieved_by_front_pair=len(retrieved),
        visual_supported_candidates=len(visual_supported),
        gate_counts=summary['gate_counts'],
        pair_pass=summary['pair_pass'],
        sequence_supported=summary['sequence_supported'],
        accepted_loops=summary['accepted_loops'],
        held_after_visual_support=[dict(
            query=r['query'], match=r['match'],
            inliers=r['sp']['inliers'], per_camera_inliers=r['sp']['per_cam'],
            translation_disagreement_m=r['sp']['diff_vs_odom_t_m'],
            rotation_disagreement_deg=r['sp']['diff_vs_odom_r_deg'],
            in_four_view_accepted=key(r) in ref_edges)
            for r in visual_supported],
        interpretation='Fixed visual-input ablation; no independent loop ground truth. '
                       'Zero accepted loops means no estimated front-pair loop '
                       'constraints reached the GNC graph under the frozen gates.')
    path = HERE / 'corrected_front_pair_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in (
        'shortlisted_candidates', 'reference_four_view_accepted',
        'reference_edges_retrieved_by_front_pair',
        'visual_supported_candidates', 'gate_counts', 'accepted_loops')}, indent=2))


if __name__ == '__main__':
    main()
