"""Dataset-independent offline entry point for multilevel closure experiments."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .floors import classify_floors
from .ground_fit import fit_ground
from .retrieval import RigRetrieval


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def json_safe(value):
    """Missing numeric evidence is explicit null, never a permissive NaN."""
    if isinstance(value,np.generic): return json_safe(value.item())
    if isinstance(value,np.ndarray): return json_safe(value.tolist())
    if isinstance(value,float) and not np.isfinite(value): return None
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value


def validate_observations(path,identity,project):
    """Reuse measured pair observations only with unchanged inputs/verifier.

    Gate, floor classification, and optimizer changes are allowed. Cached
    acceptance decisions are never reused. Source snapshots must match the
    original run's recorded hashes before the Verifier class is compared.
    """
    original=json.loads((path.parent/'provenance.json').read_text())
    for key,value in identity.items():
        if key not in ('source','settings','verified_observations') and original.get(key)!=value:
            raise ValueError('Verified observation inputs differ: '+key)
    for key in ('view_profile','mask'):
        if original['settings'][key]!=identity['settings'][key]:
            raise ValueError('Verified observation geometry setting differs: '+key)
    dependencies=['vision/src/seeker.py','vision/src/pnp_multicam.py',
                  'vision/src/match_reverse.py','eval/scripts/common.py']
    for key in dependencies:
        if original['source'][key]!=identity['source'][key]:
            raise ValueError('Verified observation dependency differs: '+key)
    relative='multifloor/verify.py'
    old_source=path.parent/'source_snapshot'/relative
    if hashlib.sha256(old_source.read_bytes()).hexdigest()!=original['source'][relative]:
        raise ValueError('Original verifier snapshot does not match its provenance')
    def verifier_ast(source):
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='edge_gate')]
        return ast.dump(tree)
    if verifier_ast(old_source.read_text())!=verifier_ast((project/relative).read_text()):
        raise ValueError('Pair verifier changed; observations must be remeasured')


def ground_metrics(poses,measurements,labels):
    per={}
    for level in sorted({x for x in labels if x is not None}):
        values=[];tilts=[]
        for i,l in enumerate(labels):
            if l!=level or not measurements[i]: continue
            m=measurements[i];normal=poses[i][:3,:3]@m['n_b']
            values.append(float(poses[i,2,3]-m['d_b']/normal[2]))
            tilts.append(float(np.degrees(np.arccos(np.clip(normal[2],-1,1)))))
        values=np.array(values)
        per[str(level)]=dict(count=len(values),center_m=float(np.median(values)),
                            std_m=float(np.std(values)),
                            p90_range_m=float(np.percentile(values,95)-np.percentile(values,5)),
                            mean_tilt_deg=float(np.mean(tilts)))
    return per


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submaps',type=Path,required=True,help='Directory with poses.json and NNNN.npy')
    parser.add_argument('--vision-input',type=Path,required=True,help='Existing vision/data with frames, masks, descriptors')
    parser.add_argument('--runtime-config',type=Path,required=True)
    parser.add_argument('--camera-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-candidates',type=Path)
    parser.add_argument('--verified-observations',type=Path,
                        help='Explicit prior pair measurements for prepare/optimize; validates inputs and verifier snapshot')
    parser.add_argument('--queries',help='Optional half-open query range START:END for staged replay')
    parser.add_argument('--same-floor-candidates',type=int,default=12)
    parser.add_argument('--max-odom-disagreement-m',type=float,default=5.,
                        help='Hold large pose corrections for independent inspection; does not prune retrieval')
    parser.add_argument('--max-odom-disagreement-deg',type=float,default=20.)
    parser.add_argument('--stage',choices=['prepare','verify','optimize','all'],default='all')
    args=parser.parse_args()
    if args.verified_observations and args.stage not in ('prepare','optimize'):
        parser.error('Prior verified observations require stage prepare or optimize')
    if args.same_floor_candidates<=0: parser.error('Candidate budget must be positive')
    if not (np.isfinite(args.max_odom_disagreement_m) and args.max_odom_disagreement_m>0
            and np.isfinite(args.max_odom_disagreement_deg) and 0<args.max_odom_disagreement_deg<=180):
        parser.error('Pose disagreement thresholds must be finite and positive')
    args.output=args.output.resolve();args.submaps=args.submaps.resolve();args.vision_input=args.vision_input.resolve()
    if (args.output==args.vision_input or args.output==args.submaps
            or args.vision_input in args.output.parents or args.submaps in args.output.parents):
        parser.error('Outputs must be outside immutable input directories')
    project=Path(__file__).resolve().parents[1]
    os.environ.update(SEEKER_RESEARCH_ROOT=str(project),SEEKER_SUBMAP_ROOT=str(args.submaps),
                      SEEKER_RUNTIME_CONFIG=str(args.runtime_config.resolve()),
                      SEEKER_CAMERA_CONFIG=str(args.camera_config.resolve()),
                      SEEKER_VISION_ROOT=str(args.output/'vision'),SEEKER_MASK='v2')
    sys.path[:0]=[str(project/'vision/src'),str(project/'eval/scripts')]
    (args.output/'vision/data').mkdir(parents=True,exist_ok=True)
    for name in ('frames','masks'):
        link=args.output/'vision/data'/name
        expected=(args.vision_input/name).resolve()
        if link.exists() and link.resolve()!=expected:
            raise ValueError('Input link points to a different dataset: '+str(link))
        if not link.exists(): link.symlink_to(expected,target_is_directory=True)
    raw_path=args.submaps/'poses.json'
    metadata=json.loads(raw_path.read_text())
    if [m['id'] for m in metadata]!=list(range(len(metadata))):
        raise ValueError('poses.json requires consecutive IDs starting at zero')
    times=np.array([m['t'] for m in metadata],float)
    poses=np.repeat(np.eye(4)[None],len(metadata),axis=0)
    poses[:,:3,3]=np.array([m['p'] for m in metadata])
    poses[:,:3,:3]=Rotation.from_quat([m['q'] for m in metadata]).as_matrix()
    descriptor_path=args.vision_input/'global_dinov2.npz'
    with np.load(descriptor_path) as data:
        descriptors=data['desc']; nodes=data['nodes'].tolist()
    if descriptors.ndim!=4 or descriptors.shape[1]!=4:
        raise ValueError('Expected descriptors[node,camera,view,dimension]')
    profile=os.environ.get('SEEKER_VIEW_PROFILE','horizontal3')
    if profile not in ('horizontal3','ring5') or descriptors.shape[2]!=(5 if profile=='ring5' else 3):
        raise ValueError('Descriptor views must match the chosen projection profile; regenerate descriptors first')
    nodes=[i for i in nodes if 0<=i<len(times)]
    # A new dataset/source/config must never silently resume old cached results.
    identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [raw_path,descriptor_path,args.runtime_config,args.camera_config]}
    identity['submaps']={str(i):hashlib.sha256((args.submaps/f'{i:04d}.npy').read_bytes()).hexdigest()
                         for i in range(len(metadata))}
    identity['source']={str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [*Path(__file__).parent.glob('*.py'),
                                  project/'vision/src/seeker.py',project/'vision/src/pnp_multicam.py',
                                  project/'vision/src/match_reverse.py',project/'vision/src/degeneracy.py',
                                  project/'eval/scripts/common.py']}
    identity['settings']=dict(same_floor_candidates=args.same_floor_candidates,queries=args.queries,
                              view_profile=os.environ.get('SEEKER_VIEW_PROFILE','horizontal3'),mask='v2',
                              max_odom_disagreement_m=args.max_odom_disagreement_m,
                              max_odom_disagreement_deg=args.max_odom_disagreement_deg)
    identity['masks']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.vision_input/'masks').glob('*.png')}
    if args.baseline_candidates:
        identity['baseline_candidates']=hashlib.sha256(args.baseline_candidates.read_bytes()).hexdigest()
    if args.verified_observations:
        args.verified_observations=args.verified_observations.resolve()
        validate_observations(args.verified_observations,identity,project)
        identity['verified_observations']=dict(path=str(args.verified_observations),
                   sha256=hashlib.sha256(args.verified_observations.read_bytes()).hexdigest())
    provenance=args.output/'provenance.json'
    if provenance.exists() and json.loads(provenance.read_text())!=identity:
        raise ValueError('Run inputs/settings/source changed; choose a new output directory')
    save(provenance,identity)
    for relative,digest in identity['source'].items():
        snapshot=args.output/'source_snapshot'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((project/relative).read_bytes())
    if args.stage in ('prepare','all'):
        measurements=[]
        for i,pose in enumerate(poses):
            measurement=fit_ground(pose,np.load(args.submaps/f'{i:04d}.npy'),seed=i)
            measurements.append(measurement)
            if i%20==0: print('floor_fit',i,len(poses),flush=True)
        floors=classify_floors(times,poses[:,:3,3],measurements)
        floors['ground_z']=[x if np.isfinite(x) else None for x in floors['ground_z']]
        save(args.output/'ground_measurements.json',measurements)
        save(args.output/'floor_hypotheses.json',floors)
        retrieval=RigRetrieval(descriptors,nodes,times,floors['labels'])
        selected=[];audit=[]
        query_nodes=nodes
        if args.queries:
            start,end=map(int,args.queries.split(':'))
            query_nodes=[i for i in nodes if start<=i<end]
        for i in query_nodes:
            rows,counts=retrieval.shortlist(i,limit=args.same_floor_candidates,unknown_limit=0)
            selected.extend(rows);audit.append(dict(query=i,history=counts,selected=len(rows)))
        save(args.output/'shortlist.json',selected)
        save(args.output/'retrieval_audit.json',audit)
        print('floor_levels',floors['levels'],'known',len(times)-len(floors['unknown_nodes']),
              'shortlist',len(selected),flush=True)
    else:
        floors=json.loads((args.output/'floor_hypotheses.json').read_text())
        measurements=json.loads((args.output/'ground_measurements.json').read_text())
    if args.stage in ('verify','all'):
        from .verify import Verifier
        existing=args.output/'candidates.json'
        records=json.loads(existing.read_text()) if existing.exists() else []
        done={(r['query'],r['match']) for r in records}
        baseline={}
        if args.baseline_candidates:
            baseline={(r['query'],r['match']):r for r in json.loads(args.baseline_candidates.read_text())}
        shortlist=json.loads((args.output/'shortlist.json').read_text())
        # Preserve baseline verified pairs as observations, then apply the same
        # new gates. Their acceptance is never copied from the baseline.
        for (i,j),r in baseline.items():
            if i>=len(times) or j>=len(times) or floors['labels'][i] is None or floors['labels'][i]!=floors['labels'][j]: continue
            if (i,j) not in done and (i,j) not in {(v['query'],v['match']) for v in shortlist}:
                shortlist.append(dict(query=i,match=j,rank=r['rank'],score=r['score'],shift=r['shift'],
                                      floor_id=floors['labels'][i],floor_relation='same',baseline_observation=True))
        verifier=Verifier(descriptors.shape,args.vision_input/'featcache/v2')
        for row in shortlist:
            i,j=row['query'],row['match']
            if (i,j) in done: continue
            similarity=np.einsum('avd,bwd->avbw',descriptors[i],descriptors[j])
            old=baseline.get((i,j),{}).get('sp',{})
            if old.get('inliers',0)<10 and old and baseline[i,j].get('shift')==row['shift']:
                record=dict(row,sp=old,reused_failed_baseline=True)
            else:
                record=dict(row,sp=verifier.verify(i,j,row['shift'],similarity))
            # Failed visual pairs need the same bounded-memory handling as
            # successful pairs across a long multilevel recording.
            if len(verifier.mr.cache)>600: verifier.mr.cache.clear()
            if len(verifier.pn._sub)>80: verifier.pn._sub.clear()
            records.append(record);done.add((i,j));save(existing,records)
            print('verified',i,j,record['sp'].get('inliers',0),len(records),'/',len(shortlist),flush=True)
    if args.stage in ('optimize','all'):
        from .verify import edge_gate
        from .sequence import supported_sequences
        from .backend import optimize
        records=json.loads((args.verified_observations or args.output/'candidates.json').read_text())
        passing=[];gate_audit=[]
        for row in records:
            decision=edge_gate(row,floors['labels'],args.max_odom_disagreement_m,args.max_odom_disagreement_deg)
            gate_audit.append(dict(query=row['query'],match=row['match'],decision=decision))
            if decision=='pair_pass': passing.append(row)
        supported=supported_sequences(passing,poses,times,floors['labels'],max_gap_sec=24.)
        save(args.output/'gate_audit.json',gate_audit)
        save(args.output/'sequence_supported.json',supported)
        corrected,result=optimize(poses,times,supported,measurements,floors)
        baseline=ground_metrics(poses,measurements,floors['labels'])
        after=ground_metrics(corrected,measurements,floors['labels'])
        violations=[]
        for level in baseline:
            if after[level]['p90_range_m']>baseline[level]['p90_range_m']+.02:
                violations.append('floor_'+level+'_ground_range_worsened')
            if after[level]['mean_tilt_deg']>baseline[level]['mean_tilt_deg']+.5:
                violations.append('floor_'+level+'_tilt_worsened')
        for tr in floors['transitions']:
            delta=result['floor_heights'][tr['to_level']]-result['floor_heights'][tr['from_level']]
            if abs(delta-tr['delta_m'])>max(.8,3*tr['sigma_m']): violations.append('connection_height_changed')
            if abs(delta)<1.8: violations.append('levels_collapsed')
        if not np.isfinite(corrected).all(): violations.append('nonfinite_trajectory')
        result.update(raw_ground_metrics=baseline,optimized_ground_metrics=after,
                      stage_counts=dict(Counter(x['decision'] for x in gate_audit)),
                      sequence_supported=len(supported),quality_violations=violations,
                      release_ok=not violations,
                      limitation='Internal consistency only; this does not establish wall thickness or surveyed floor spacing accuracy.')
        save(args.output/'summary.json',result)
        table=np.column_stack([times,poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()])
        np.savetxt(args.output/'raw_poses.txt',table,fmt='%.9f')
        table=np.column_stack([times,corrected[:,:3,3],Rotation.from_matrix(corrected[:,:3,:3]).as_quat()])
        np.savetxt(args.output/('optimized_poses.txt' if not violations else 'held_hypothesis_poses.txt'),table,fmt='%.9f')
        print(json.dumps(result,indent=2),flush=True)
        if violations: raise SystemExit(2)


if __name__=='__main__':
    main()
