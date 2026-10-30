"""Front-camera-pair end-to-end visual-input ablation on four-floor flight.

c0+c1 are used on both traversals. LiDAR ICP, sequence support, ground
factors and GNC are unchanged. This is not a vision-only SLAM benchmark.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


BASE = Path('/home/nuc/omnilivo_floor_loop_trial/four_floor_20261013_160929')
REPO = Path('/home/nuc/five_floor_optimization_20261013')
SUB = Path(str(BASE) + '_submaps')
RELEASE = BASE / 'loop_latest_20261014'
WORK = BASE / 'front_pair_c0_c1_20261021'
SOURCE = REPO / 'research/multifloor/verify.py'
EXPERIMENT = Path(__file__).resolve().parent


def setup(vision):
    os.environ.update(
        SEEKER_RESEARCH_ROOT=str(REPO / 'research'),
        SEEKER_SUBMAP_ROOT=str(SUB),
        SEEKER_RUNTIME_CONFIG=str(BASE / 'loop_inputs/runtime.yaml'),
        SEEKER_CAMERA_CONFIG=str(REPO / 'mapping/src/Omni-LIVO/config/seeker_raw_cam.yaml'),
        SEEKER_VISION_ROOT=str(vision), SEEKER_MASK='v2',
        OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
        TORCH_HOME='/home/nuc/.cache/torch')
    sys.path[:0] = [str(REPO), str(REPO / 'research/vision/src'),
                    str(REPO / 'research/eval/scripts')]


def save(path, obj):
    from research.multifloor.run import save as original_save
    original_save(path, obj)


def mono_source():
    source = SOURCE.read_text()
    replacements = [
        # pnp_multicam.py parses filenames by splitting on '_c'. The tag must
        # not contain that delimiter or both camera files get misidentified.
        ("tag = 'multifloor_v1'", "tag = 'frontpair01'"),
        ('for ci in range(4):', 'for ci in (0,1):'),
        ('cj = (ci+shift)%4', 'cj = (ci+shift)%2'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected verifier source for replacement: {old}')
        source = source.replace(old, new)
    return source


def load_mono_module():
    path = WORK / 'verify_mono.py'
    if path.read_text() != mono_source():
        raise RuntimeError('Monocular verifier no longer matches frozen source')
    spec = importlib.util.spec_from_file_location('research.multifloor.verify_mono', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    setup(WORK / 'prep_vision')
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'verify_mono.py').write_text(mono_source())
    (WORK / 'backend.py').write_bytes((EXPERIMENT / 'backend.py').read_bytes())
    from research.multifloor.retrieval import RigRetrieval
    from research.multifloor.floors import floor_relation
    from collections import Counter
    with np.load(BASE / 'vision_input/global_dinov2.npz') as z:
        desc, nodes = z['desc'], z['nodes']
    meta = json.loads((SUB / 'poses.json').read_text())
    times = np.array([m['t'] for m in meta], float)
    labels = json.loads((RELEASE / 'floor_hypotheses.json').read_text())['labels']
    assert len(desc) == len(meta) == len(labels)
    rr = RigRetrieval(desc, nodes, times, labels)
    tasks = []
    for i in rr.nodes:
        ranked = []
        for j in rr.nodes:
            if (i, j) not in rr.cp or floor_relation(i, j, labels) != 'same':
                continue
            mean, std = rr.level_stats.get(labels[i], (rr.mean, rr.std))
            cp = (rr.cp[i, j] - mean) / std
            pair_scores = [float((cp[0, s] + cp[1, (1+s)%2]) / 2) for s in (0, 1)]
            shift = int(np.argmax(pair_scores))
            ranked.append((pair_scores[shift], j, shift))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        bucket = Counter()
        count = 0
        for score, j, shift in ranked:
            bin_ = int((times[j] - times[0]) // 12)
            if bucket[bin_] >= 2:
                continue
            tasks.append(dict(query=int(i), match=int(j), score=float(score),
                              rank=count+1, shift=shift, camera=4))
            bucket[bin_] += 1
            count += 1
            if count >= 12:
                break
    assert len(tasks) == 1100, len(tasks)
    short_path = WORK / 'shortlist.json'
    save(short_path, tasks)
    worker_count = 8
    for w in range(worker_count):
        folder = WORK / 'workers' / str(w)
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'tasks.json', tasks[w::worker_count])
        data = folder / 'vision/data'
        data.mkdir(parents=True, exist_ok=True)
        for name in ('frames', 'masks'):
            link = data / name
            expected = (BASE / 'vision_input' / name).resolve()
            if link.exists() and link.resolve() != expected:
                raise RuntimeError('Input link differs: ' + str(link))
            if not link.exists():
                link.symlink_to(expected, target_is_directory=True)
    files = [SOURCE, short_path, EXPERIMENT / 'backend.py',
             BASE / 'vision_input/global_dinov2.npz', SUB / 'poses.json',
             REPO / 'research/vision/src/pnp_multicam.py',
             REPO / 'research/vision/src/match_reverse.py']
    save(WORK / 'provenance.json', dict(
        description=__doc__, worker_count=worker_count,
        cameras=[0, 1], tasks=len(tasks),
        files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        verifier_patch=dict(front_pair_permutation=True, two_camera_gate=True,
                            all_other_thresholds_unchanged=True)))
    print('PREPARED', WORK, len(tasks), flush=True)


def worker(w):
    folder = WORK / 'workers' / str(w)
    setup(folder / 'vision')
    import torch
    import cv2
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    module = load_mono_module()
    with np.load(BASE / 'vision_input/global_dinov2.npz') as z:
        desc = z['desc']
    verifier = module.Verifier(desc.shape, BASE / 'vision_input/featcache/v2')
    torch.set_num_threads(2)
    tasks = json.loads((folder / 'tasks.json').read_text())
    output = folder / 'candidates.json'
    records = json.loads(output.read_text()) if output.exists() else []
    done = {(r['camera'], r['query'], r['match']) for r in records}
    start = time.monotonic()
    for row in tasks:
        cam, i, j = row['camera'], row['query'], row['match']
        if (cam, i, j) in done:
            continue
        verifier.camera = cam
        similarity = np.einsum('avd,bwd->avbw', desc[i], desc[j])
        record = dict(row, sp=verifier.verify(i, j, row['shift'], similarity))
        records.append(record)
        done.add((cam, i, j))
        save(output, records)
        if len(verifier.mr.cache) > 600:
            verifier.mr.cache.clear()
        if len(verifier.pn._sub) > 80:
            verifier.pn._sub.clear()
        if len(records) % 10 == 0:
            print('PROGRESS', w, len(records), '/', len(tasks), 'camera', cam,
                  'elapsed_s', round(time.monotonic() - start), flush=True)
    save(folder / 'complete.json', dict(worker=w, records=len(records),
                                       elapsed_seconds=time.monotonic() - start))
    print('COMPLETE', w, len(records), flush=True)


def load_backend():
    spec = importlib.util.spec_from_file_location('mono_ablation_backend', WORK / 'backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(only_camera=None):
    setup(WORK / 'prep_vision')
    from research.multifloor.sequence import supported_sequences
    from research.multifloor.run import ground_metrics
    module = load_mono_module()
    backend = load_backend()
    tasks = sum((json.loads((WORK / 'workers' / str(w) / 'tasks.json').read_text())
                 for w in range(8)), [])
    records = sum((json.loads((WORK / 'workers' / str(w) / 'candidates.json').read_text())
                   for w in range(8)), [])
    if only_camera is None:
        assert len(tasks) == len(records) == 1100
    task_keys = {(r['camera'], r['query'], r['match']) for r in tasks}
    record_keys = {(r['camera'], r['query'], r['match']) for r in records}
    if only_camera is None:
        assert len(task_keys) == len(record_keys) == 1100 and task_keys == record_keys
    else:
        selected_tasks = {key for key in task_keys if key[0] == only_camera}
        selected_records = {key for key in record_keys if key[0] == only_camera}
        assert len(selected_tasks) == len(selected_records) == 1100
        assert selected_tasks == selected_records
    meta = json.loads((SUB / 'poses.json').read_text())
    times = np.array([m['t'] for m in meta], float)
    poses = np.repeat(np.eye(4)[None], len(meta), axis=0)
    poses[:, :3, 3] = [m['p'] for m in meta]
    poses[:, :3, :3] = Rotation.from_quat([m['q'] for m in meta]).as_matrix()
    floors = json.loads((RELEASE / 'floor_hypotheses.json').read_text())
    measurements = json.loads((RELEASE / 'ground_measurements.json').read_text())
    before = ground_metrics(poses, measurements, floors['labels'])
    full = json.loads((BASE / 'sequence_gap_ablation_20261014/gap_24_summary.json').read_text())
    full_edges = {(e['i'], e['j']) for e in full['accepted_edges'] if e['gnc_weight'] > .5}
    ordinal = {level: idx + 1 for idx, level in enumerate(np.argsort(floors['levels']).tolist())}
    summary = {}
    for cam in (4,) if only_camera is None else (only_camera,):
        start = time.monotonic()
        rows = [r for r in records if r['camera'] == cam]
        assert len(rows) == 1100
        gate = { (r['query'], r['match']): module.edge_gate(r, floors['labels']) for r in rows }
        passing = [r for r in rows if gate[r['query'], r['match']] == 'pair_pass']
        supported = supported_sequences(passing, poses, times, floors['labels'], max_gap_sec=24.)
        corrected, result = backend.optimize(poses, times, supported, measurements, floors)
        after = ground_metrics(corrected, measurements, floors['labels'])
        violations = []
        for level in before:
            if after[level]['p90_range_m'] > before[level]['p90_range_m'] + .02:
                violations.append('floor_' + level + '_ground_range_worsened')
            if after[level]['mean_tilt_deg'] > before[level]['mean_tilt_deg'] + .5:
                violations.append('floor_' + level + '_tilt_worsened')
        for tr in floors['transitions']:
            delta = result['floor_heights'][tr['to_level']] - result['floor_heights'][tr['from_level']]
            if abs(delta - tr['delta_m']) > max(.8, 3 * tr['sigma_m']):
                violations.append('connection_height_changed')
            if abs(delta) < 1.8:
                violations.append('levels_collapsed')
        if not np.isfinite(corrected).all():
            violations.append('nonfinite_trajectory')
        accepted = [e for e in result['edges'] if e['gnc_weight'] > .5]
        actual = {(e['i'], e['j']) for e in accepted}
        out = dict(
            method='front_pair_c0_c1_end_to_end_visual_input',
            explanation='Front cameras c0 and c1 alone are used in both visits, from retrieval through geometric verification and LiDAR-depth PnP. Other factors and thresholds match the four-view reference.',
            candidates=len(rows), gate_counts=dict(Counter(gate.values())),
            pair_pass=len(passing), sequence_supported=len(supported),
            accepted_loops=result['loop_count'],
            accepted_by_floor={str(f):sum(ordinal[e['floor']] == f for e in accepted) for f in range(1, 5)},
            accepted_edges=accepted, accepted_overlap_full=len(actual & full_edges),
            accepted_unique=len(actual - full_edges), full_unique=len(full_edges - actual),
            ground_before=before, ground_after=after, quality_violations=violations,
            floor_heights=result['floor_heights'],
            elapsed_evaluation_seconds=time.monotonic() - start)
        save(WORK / f'camera_c{cam}_summary.json', out)
        save(WORK / f'camera_c{cam}_gate_audit.json', [dict(camera=cam, query=r['query'], match=r['match'], decision=gate[r['query'], r['match']]) for r in rows])
        save(WORK / f'camera_c{cam}_sequence_supported.json', supported)
        np.savetxt(WORK / f'camera_c{cam}_optimized_poses.txt',
                   np.column_stack([times, corrected[:, :3, 3],
                                    Rotation.from_matrix(corrected[:, :3, :3]).as_quat()]), fmt='%.9f')
        summary[f'c{cam}'] = {k:out[k] for k in ('candidates', 'gate_counts',
                    'pair_pass', 'sequence_supported', 'accepted_loops',
                    'accepted_by_floor', 'accepted_overlap_full', 'accepted_unique',
                    'quality_violations')}
        print('FINAL', cam, json.dumps(summary[f'c{cam}']), flush=True)
    save(WORK / ('summary_all.json' if only_camera is None else f'summary_c{only_camera}.json'), summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'worker', 'evaluate'])
    parser.add_argument('--worker', type=int)
    parser.add_argument('--camera', type=int, choices=range(5))
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'worker':
        if args.worker is None or not 0 <= args.worker < 8:
            parser.error('--worker must be 0..7')
        worker(args.worker)
    else: evaluate(args.camera)


if __name__ == '__main__': main()
