import sys, unittest
from pathlib import Path
import numpy as np
try:
    import gtsam
except ImportError:
    gtsam=None
from research.multifloor.backend import floor_error,height_difference,optimize,loop_information
research=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(research/'vision/src'),str(research/'eval/scripts')]

@unittest.skipIf(gtsam is None,'GTSAM is required for backend checks')
class BackendTests(unittest.TestCase):
    def test_floor_factor_height_and_tilt_jacobians(self):
        values=gtsam.Values();pose=gtsam.Pose3(gtsam.Rot3.RzRyRx(.05,-.03,1.),np.array([2.,3.,4.]))
        values.insert(0,pose);height=gtsam.symbol('h',0);values.insert(height,2.2)
        normal=np.array([0.,0.,1.]);jac=[None,None]
        residual=floor_error(normal,1.8,0,height,None,values,jac)
        self.assertEqual(jac[0].shape,(3,6));self.assertEqual(jac[1].shape,(3,1))
        step=np.array([.004,-.003,.002,.003,.002,-.005]);dh=.002
        next_values=gtsam.Values();next_values.insert(0,pose.retract(step));next_values.insert(height,2.2+dh)
        actual=floor_error(normal,1.8,0,height,None,next_values,None)
        np.testing.assert_allclose(actual,residual+jac[0]@step+jac[1].ravel()*dh,atol=5e-5)
    def test_separate_planes_keep_floor_spacing(self):
        poses=np.repeat(np.eye(4)[None],40,axis=0);poses[:,0,3]=np.arange(40)
        poses[:20,2,3]=1.8+np.linspace(0,.3,20);poses[20:,2,3]=5.1+np.linspace(0,-.3,20)
        measures=[dict(n_b=[0.,0.,1.],d_b=1.8) for _ in poses]
        floors=dict(levels=[.15,3.15],labels=[0]*20+[1]*20,
                    transitions=[dict(from_level=0,to_level=1,delta_m=3.,sigma_m=.4)])
        result,audit=optimize(poses,np.arange(40)*3.,[],measures,floors)
        self.assertEqual(audit['loop_count'],0)
        self.assertGreater(audit['floor_heights'][1]-audit['floor_heights'][0],2.8)
        for start in [0,20]:
            self.assertLess(np.std(result[start:start+20,2,3]),np.std(poses[start:start+20,2,3]))
    def test_covariance_frame_conversion_and_degenerate_axis(self):
        transform=np.eye(4);transform[:3,:3]=gtsam.Rot3.Rz(np.pi/2).matrix();transform[:3,3]=[1.,2.,0.]
        edge=dict(T=transform,cov=np.diag([.001]*3+[.01,.02,.03]),
                  deg=dict(degenerate=True,weak_dir_j=[1.,0.,0.]))
        info=loop_information(edge);cov=np.linalg.inv(info)
        self.assertGreaterEqual(cov[4,4],.25-1e-10)
        self.assertGreater(np.linalg.eigvalsh(info).min(),0.)

if __name__=='__main__': unittest.main(verbosity=2)
