import unittest
import numpy as np
from .floors import classify_floors,floor_relation
from .retrieval import RigRetrieval
from .sequence import supported_sequences,consistent
from .verify import edge_gate
from .ground_fit import fit_ground


def measure(z):
    return dict(inliers=300,tilt_deg=0.,rmse=.01,z_floor=z,n_b=[0.,0.,1.],d_b=1.8)


class FloorTests(unittest.TestCase):
    def test_five_levels_and_return_keep_distinct_heights(self):
        heights=np.r_[np.repeat(np.arange(5)*3.3,20),np.repeat(0.,20)]
        positions=np.c_[np.tile(np.arange(20),6),np.zeros(len(heights)),heights+1.8]
        floors=classify_floors(np.arange(len(heights))*3.,positions,[measure(h) for h in heights])
        self.assertEqual(len(floors['levels']),5)
        self.assertEqual(floors['labels'][8],floors['labels'][108])
        self.assertEqual(floor_relation(8,48,floors['labels']),'different')
        self.assertIsNone(floors['labels'][19])  # boundary cannot certify a level

    def test_missing_ground_does_not_certify_floor(self):
        positions=np.c_[np.arange(40),np.zeros(40),np.ones(40)]
        measurements=[measure(0.)]*40
        measurements[15:25]=[None]*10
        floors=classify_floors(np.arange(40)*3.,positions,measurements)
        self.assertTrue(all(floors['labels'][i] is None for i in range(15,25)))
        self.assertEqual(floor_relation(10,20,floors['labels']),'unknown')

    def test_short_stair_landing_is_not_a_floor(self):
        times=np.arange(5)*3.
        positions=np.c_[np.arange(5)*.1,np.zeros(5),np.ones(5)]
        floors=classify_floors(times,positions,[measure(1.6)]*5)
        self.assertEqual(floors['levels'],[])

    def test_vertical_motion_cannot_supply_horizontal_travel(self):
        positions=np.c_[np.zeros(20),np.zeros(20),np.arange(20)]
        floors=classify_floors(np.arange(20)*3.,positions,[measure(0.)]*20)
        self.assertEqual(floors['levels'],[])

    def test_empty_data_and_invalid_time_fail_explicitly(self):
        with self.assertRaises(ValueError):
            classify_floors([3.,3.],np.zeros((2,3)),[measure(0.)]*2)


class RetrievalTests(unittest.TestCase):
    def test_cross_floor_impostors_cannot_use_same_floor_budget(self):
        rng=np.random.default_rng(4)
        desc=rng.normal(size=(100,4,3,16))
        desc/=np.linalg.norm(desc,axis=-1,keepdims=True)
        desc[99]=desc[7]
        for i in range(10,80): desc[i]=desc[99]
        labels=[0]*100
        for i in range(10,80): labels[i]=1
        retrieval=RigRetrieval(desc,range(100),np.arange(100)*3.,labels)
        shortlist,audit=retrieval.shortlist(99,limit=4,unknown_limit=0)
        self.assertIn(7,[r['match'] for r in shortlist])
        self.assertTrue(all(r['floor_relation']=='same' for r in shortlist))
        self.assertGreater(audit['different'],60)
        self.assertTrue(all(np.isfinite(r['score']) for r in shortlist))

    def test_invalid_and_zero_descriptors_excluded(self):
        desc=np.ones((20,4,3,8))/np.sqrt(8)
        desc[4]=0;desc[7]=np.nan
        r=RigRetrieval(desc,range(20),np.arange(20)*3.,[0]*20)
        self.assertNotIn(4,r.nodes);self.assertNotIn(7,r.nodes)


class SequenceTests(unittest.TestCase):
    def fixture(self):
        poses=np.repeat(np.eye(4)[None],40,axis=0)
        poses[:20,0,3]=np.arange(20)*2.
        poses[20:,0,3]=np.arange(20)[::-1]*2.
        rows=[dict(query=20+k,match=19-k,sp={'icp_T':np.eye(4).tolist(),'inliers':30}) for k in range(8)]
        return poses,rows,np.arange(40)*3.,[0]*40

    def test_reverse_sequence_is_supported(self):
        p,r,t,l=self.fixture()
        good=supported_sequences(r,p,t,l)
        self.assertEqual(len(good),8)
        self.assertTrue(all(x['sequence_direction']==-1 for x in good))

    def test_same_target_cannot_be_counted_as_distinct_witnesses(self):
        p,r,t,l=self.fixture()
        for x in r: x['match']=19
        self.assertEqual(supported_sequences(r,p,t,l),[])

    def test_other_floor_sequence_cannot_pass(self):
        p,r,t,l=self.fixture();l[20:]=[1]*20
        self.assertEqual(supported_sequences(r,p,t,l),[])

    def test_stationary_jitter_cannot_supply_six_meter_span(self):
        p,r,t,l=self.fixture();p[:,:3,3]=0
        self.assertEqual(supported_sequences(r,p,t,l),[])

    def test_complete_link_rejects_transitive_borrowing(self):
        a,b,c=np.eye(4),np.eye(4),np.eye(4)
        b[0,3]=.5;c[0,3]=1.
        self.assertTrue(consistent(a,b));self.assertTrue(consistent(b,c))
        self.assertFalse(consistent(a,c))


class GateTests(unittest.TestCase):
    def row(self):
        return dict(query=2,match=0,sp=dict(inliers=30,per_cam={'0':10,'1':10,'2':10},
          icp_fitness=.8,icp_corr_t=.1,icp_corr_r=1.,icp_rmse=.03,reverse_fitness=.7,
          reverse_rmse=.04,icp_T=np.eye(4).tolist(),diff_vs_odom_t_m=.3,diff_vs_odom_r_deg=1.))

    def test_cross_and_unknown_floor_are_held(self):
        self.assertEqual(edge_gate(self.row(),[0,0,1]),'reject_cross_floor')
        self.assertEqual(edge_gate(self.row(),[0,0,None]),'hold_unknown_floor')

    def test_asymmetric_registration_is_rejected(self):
        row=self.row();row['sp']['reverse_fitness']=.1
        self.assertEqual(edge_gate(row,[0,0,0]),'reject_symmetric_geometry')

    def test_invalid_transform_is_rejected(self):
        row=self.row();row['sp']['icp_T'][0][0]=2.
        self.assertEqual(edge_gate(row,[0,0,0]),'reject_invalid_transform')

    def test_good_observation_still_requires_later_sequence(self):
        self.assertEqual(edge_gate(self.row(),[0,0,0]),'pair_pass')

    def test_two_supported_cameras_pass_but_one_camera_does_not(self):
        row=self.row();row['sp']['per_cam']={'2':20,'3':10}
        self.assertEqual(edge_gate(row,[0,0,0]),'pair_pass')
        row['sp']['per_cam']={'2':30}
        self.assertEqual(edge_gate(row,[0,0,0]),'reject_visual_support')

    def test_large_repetitive_corridor_alias_is_held(self):
        row=self.row();row['sp']['diff_vs_odom_t_m']=48.
        self.assertEqual(edge_gate(row,[0,0,0]),'hold_large_pose_disagreement')

    def test_missing_or_nonfinite_geometry_is_rejected(self):
        for value in [None,float('nan'),float('inf')]:
            row=self.row();row['sp']['icp_rmse']=value
            self.assertEqual(edge_gate(row,[0,0,0]),'reject_symmetric_geometry')
        row['sp']['diff_vs_odom_t_m']=float('nan')
        self.assertEqual(edge_gate(row,[0,0,0]),'hold_large_pose_disagreement')

    def test_ground_fit_is_reproducible(self):
        rng=np.random.default_rng(7)
        cloud=np.c_[rng.uniform(-4,4,(1000,2)),rng.normal(-1.8,.01,1000)]
        a=fit_ground(np.eye(4),cloud,seed=9);b=fit_ground(np.eye(4),cloud,seed=9)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['d_b'],1.8,places=2)


if __name__=='__main__': unittest.main()
