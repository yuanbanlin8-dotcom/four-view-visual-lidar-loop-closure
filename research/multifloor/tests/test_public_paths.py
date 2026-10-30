"""Public entry-point paths must reach the camera and LiDAR helpers."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class PublicPathTests(unittest.TestCase):
    def test_camera_and_submap_overrides(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            env=dict(os.environ, SEEKER_VISION_ROOT=str(p/'vision'),
                     SEEKER_SUBMAP_ROOT=str(p/'submaps'),
                     SEEKER_CAMERA_CONFIG=str(p/'camera.yaml'),
                     SEEKER_RUNTIME_CONFIG=str(p/'runtime.yaml'))
            code="""
import sys,os
from pathlib import Path
sys.path[:0]=[sys.argv[1]+'/vision/src',sys.argv[1]+'/eval/scripts']
import seeker,common
assert seeker.ROOT==Path(os.environ['SEEKER_VISION_ROOT'])
assert seeker.KF==Path(os.environ['SEEKER_SUBMAP_ROOT'])
assert seeker.CAMCFG==Path(os.environ['SEEKER_CAMERA_CONFIG'])
assert seeker.RUNTIME==Path(os.environ['SEEKER_RUNTIME_CONFIG'])
assert common.SUBMAPS==os.environ['SEEKER_SUBMAP_ROOT']
"""
            subprocess.run([sys.executable,'-c',code,str(root)],env=env,check=True)
