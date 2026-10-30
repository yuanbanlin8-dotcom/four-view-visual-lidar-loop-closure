import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research.multifloor.run import validate_observations,json_safe


class ObservationReuseTests(unittest.TestCase):
    def test_nonfinite_legacy_evidence_is_explicitly_missing(self):
        self.assertEqual(json_safe({'rmse':float('nan'),'other':[float('inf'),.2]}),
                         {'rmse':None,'other':[None,.2]})

    def test_numpy_optimizer_results_remain_json_serializable(self):
        value={'count':np.int64(2),'accepted':np.bool_(True),'weight':np.array([np.nan,.2])}
        self.assertEqual(json.loads(json.dumps(json_safe(value),allow_nan=False)),
                         {'count':2,'accepted':True,'weight':[None,.2]})
    def fixture(self,folder):
        folder=Path(folder);project=folder/'project';run=folder/'original'
        verify=project/'multifloor/verify.py';verify.parent.mkdir(parents=True)
        old='import numpy as np\nclass Verifier:\n    def verify(self): return 1\ndef edge_gate(row,labels): return 3\n'
        verify.write_text(old.replace('return 3','return 2'))
        snapshot=run/'source_snapshot/multifloor/verify.py';snapshot.parent.mkdir(parents=True)
        snapshot.write_text(old)
        source={name:'same' for name in ['vision/src/seeker.py','vision/src/pnp_multicam.py',
                                        'vision/src/match_reverse.py','eval/scripts/common.py']}
        source['multifloor/verify.py']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        identity={'submaps':{'0':'data'},'masks':{'mask':'data'},'source':source,
                  'settings':{'view_profile':'horizontal3','mask':'v2'}}
        (run/'provenance.json').write_text(json.dumps(identity))
        return project,run/'candidates.json',identity,verify,snapshot

    def test_gate_change_reuses_pair_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            project,path,identity,_,_=self.fixture(folder)
            validate_observations(path,identity,project)

    def test_changed_inputs_and_verifier_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            project,path,identity,verify,_=self.fixture(folder)
            identity['submaps']={'0':'different'}
            with self.assertRaises(ValueError):validate_observations(path,identity,project)
            identity['submaps']={'0':'data'}
            verify.write_text(verify.read_text().replace('return 1','return 4'))
            with self.assertRaises(ValueError):validate_observations(path,identity,project)

    def test_changed_snapshot_cannot_claim_old_verifier(self):
        with tempfile.TemporaryDirectory() as folder:
            project,path,identity,_,snapshot=self.fixture(folder)
            snapshot.write_text('class Verifier: pass')
            with self.assertRaises(ValueError):validate_observations(path,identity,project)


if __name__=='__main__':unittest.main()
