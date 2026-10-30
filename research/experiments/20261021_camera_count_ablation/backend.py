"""GNC-weighted loop factors and independent latent floor heights."""
from functools import partial
from collections import defaultdict
import numpy as np


def adjoint(transform):
    rotation,translation = transform[:3,:3],transform[:3,3]
    x,y,z = translation
    skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    value = np.zeros((6,6))
    value[:3,:3] = value[3:,3:] = rotation
    value[3:,:3] = skew @ rotation
    return value


def loop_information(edge):
    covariance = edge.get('cov')
    if covariance is None:
        covariance = np.diag([np.radians(2)**2]*3+[.2**2]*3)
    else:
        # ICP Hessian uses left increments in target j. Pose3 BetweenFactor
        # residual uses right increments in source i; convert all cross terms.
        a = adjoint(np.linalg.inv(edge['T']))
        covariance = a @ covariance @ a.T
    covariance = (covariance+covariance.T)/2
    covariance += np.diag([np.radians(2)**2]*3+[.1**2]*3)
    if edge['deg'].get('degenerate'):
        axis = edge['T'][:3,:3].T @ np.asarray(edge['deg']['weak_dir_j'])
        axis /= np.linalg.norm(axis)
        variance = axis @ covariance[3:,3:] @ axis
        covariance[3:,3:] += max(.5**2-variance,0)*np.outer(axis,axis)
    eigenvalues,vectors = np.linalg.eigh(covariance)
    covariance = vectors @ np.diag(np.maximum(eigenvalues,1e-8)) @ vectors.T
    return np.linalg.inv(covariance)*edge.get('correlation_weight',1.)


def spatial_anchors(rows, poses, times, radius=3.):
    selected=[]
    for row in sorted(rows,key=lambda r:(-r['sp']['inliers'],r['query'],r['match'])):
        i,j=row['query'],row['match']
        if any(abs(times[i]-times[a['query']])<12 and abs(times[j]-times[a['match']])<12
               and np.linalg.norm(poses[i][:3,3]-poses[a['query']][:3,3])<radius
               and np.linalg.norm(poses[j][:3,3]-poses[a['match']][:3,3])<radius for a in selected):
            continue
        selected.append(row)
    return selected


def floor_error(normal,offset,pose_key,height_key,this,values,jacobians):
    def evaluate(pose,height):
        n=pose.rotation().matrix()@normal
        # Plane intersection directly below the sensor, not perpendicular-foot Z.
        if abs(n[2])<.2:
            return np.array([n[0],n[1],100.])
        return np.array([n[0],n[1],pose.translation()[2]-offset/n[2]-height])
    pose,height=values.atPose3(pose_key),values.atDouble(height_key)
    residual=evaluate(pose,height)
    if jacobians is not None:
        jac=np.empty((3,6)); epsilon=1e-6
        for k in range(6):
            delta=np.zeros(6);delta[k]=epsilon
            jac[:,k]=(evaluate(pose.retract(delta),height)-evaluate(pose.retract(-delta),height))/(2*epsilon)
        jacobians[0]=jac
        jacobians[1]=np.array([[0.],[0.],[-1.]])
    return residual


def height_difference(first,second,delta,this,values,jacobians):
    if jacobians is not None:
        jacobians[0]=np.array([[-1.]])
        jacobians[1]=np.array([[1.]])
    return np.array([values.atDouble(second)-values.atDouble(first)-delta])


def optimize(poses,times,rows,measurements,floors,use_ground=True):
    import gtsam
    import degeneracy
    anchors=spatial_anchors(rows,poses,times)
    edges=[]
    for row in anchors:
        i,j=row['query'],row['match'];transform=np.asarray(row['sp']['icp_T'])
        deg=degeneracy.analyse(i,j,transform,poses[j][:3,:3])
        if deg['cov_6x6'] is None or not np.isfinite(deg['cov_6x6']).all():
            continue
        edge=dict(i=i,j=j,T=transform,cov=np.asarray(deg['cov_6x6']),deg=deg,row=row)
        edges.append(edge)
    loops=edges
    counts=defaultdict(int)
    for e in loops:
        counts[(floors['labels'][e['i']],e['row']['sequence_id'])]+=1
    for e in loops:
        e['correlation_weight']=min(1.,3./counts[(floors['labels'][e['i']],e['row']['sequence_id'])])
    infos=[loop_information(e) for e in loops]
    initial=gtsam.Values()
    for i,t in enumerate(poses): initial.insert(i,gtsam.Pose3(t))
    levels=floors['levels'] if use_ground else []
    height_keys=[gtsam.symbol('h',i) for i in range(len(levels))]
    for key,height in zip(height_keys,levels): initial.insert(key,float(height))
    odometry=gtsam.noiseModel.Diagonal.Sigmas(np.array([.01]*3+[.05]*3))
    ground_base=gtsam.noiseModel.Diagonal.Sigmas(np.array([np.radians(1.5)]*2+[.08]))
    ground_noise=gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),ground_base)
    def build(weights):
        graph=gtsam.NonlinearFactorGraph()
        graph.add(gtsam.PriorFactorPose3(0,gtsam.Pose3(poses[0]),gtsam.noiseModel.Isotropic.Sigma(6,1e-4)))
        for i in range(1,len(poses)):
            graph.add(gtsam.BetweenFactorPose3(i-1,i,gtsam.Pose3(np.linalg.inv(poses[i-1])@poses[i]),odometry))
        for level,key in enumerate(height_keys):
            graph.add(gtsam.PriorFactorDouble(key,float(levels[level]),gtsam.noiseModel.Isotropic.Sigma(1,.5)))
        if use_ground:
            for i,level in enumerate(floors['labels']):
                if level is None or not measurements[i]: continue
                m=measurements[i]
                graph.add(gtsam.CustomFactor(ground_noise,[i,height_keys[level]],
                    partial(floor_error,np.asarray(m['n_b']),m['d_b'],i,height_keys[level])))
            # Connection deltas are measured; equal storey spacing is never imposed.
            for tr in floors['transitions']:
                first,second=height_keys[tr['from_level']],height_keys[tr['to_level']]
                noise=gtsam.noiseModel.Isotropic.Sigma(1,tr['sigma_m'])
                graph.add(gtsam.CustomFactor(noise,[first,second],partial(height_difference,first,second,tr['delta_m'])))
        for e,info,w in zip(loops,infos,weights):
            if w <= 0:
                continue
            graph.add(gtsam.BetweenFactorPose3(e['j'],e['i'],gtsam.Pose3(e['T']),
                                              gtsam.noiseModel.Gaussian.Information(info*max(float(w),1e-8))))
        return graph
    params=gtsam.LevenbergMarquardtParams();params.setMaxIterations(100)
    def solve(graph,start):
        return gtsam.LevenbergMarquardtOptimizer(graph,start,params).optimize()
    weights=np.ones(len(loops))
    estimate=solve(build(weights),initial)
    def residuals(value):
        output=[]
        for e,info in zip(loops,infos):
            actual=value.atPose3(e['j']).between(value.atPose3(e['i']))
            residual=gtsam.Pose3.Logmap(gtsam.Pose3(e['T']).between(actual))
            output.append(float(residual@info@residual))
        return np.array(output)
    if loops:
        cutoff=16.81;res=residuals(estimate)
        mu=max(cutoff/max(2*max(res)-cutoff,1e-6),1e-4)
        for _ in range(60):
            lower,upper=mu/(mu+1)*cutoff,(mu+1)/mu*cutoff
            weights=np.where(res<=lower,1.,np.where(res>=upper,0.,np.sqrt(cutoff*mu*(mu+1)/np.maximum(res,1e-12))-mu))
            estimate=solve(build(weights),estimate);res=residuals(estimate)
            if np.all((weights<1e-3)|(weights>1-1e-3)) and mu>50: break
            mu*=1.4
    # Accepted export physically removes rejected factors rather than leaving
    # fractional/outlier constraints in the published trajectory.
    weights=(weights>.5).astype(float)
    estimate=solve(build(weights),estimate)
    final=np.array([estimate.atPose3(i).matrix() for i in range(len(poses))])
    audit=[dict(i=e['i'],j=e['j'],floor=floors['labels'][e['i']],gnc_weight=float(w),
                sequence_id=e['row']['sequence_id'],degenerate=e['deg']['degenerate'],
                correlation_weight=e['correlation_weight']) for e,w in zip(loops,weights)]
    heights=[estimate.atDouble(k) for k in height_keys]
    return final,dict(edges=audit,floor_heights=heights,ground_enabled=use_ground,
                      spatial_anchor_count=len(anchors),loop_count=int(sum(w>.5 for w in weights)))
