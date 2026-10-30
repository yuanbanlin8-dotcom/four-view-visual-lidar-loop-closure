# Reproduction guide

## Check the released results

From the repository root:

```bash
python -m pip install numpy scipy
python tools/check_release.py
python tools/rescore_ncd.py
python -m unittest research.multifloor.test_multifloor research.multifloor.tests.test_backend research.multifloor.tests.test_observation_reuse -v
```

The first script checks file hashes, final floor-height calculations and the 26-loop floor distribution. The second re-scores the included NCD candidates, constraints and trajectories into `outputs/ncd_rescore/` and verifies the published counts. Backend unit tests require GTSAM; the runner reports skips when it is unavailable.

## Run the multilevel pipeline with prepared inputs

Use a Linux environment for the ROS frontend and symlinks created by the offline entry point. The source environment dependency record is `research/requirements.txt`; installation of that complete environment has not been checked on this release workstation.

```bash
python -m research.multifloor.run \
  --submaps /absolute/path/to/submaps \
  --vision-input /absolute/path/to/vision/data \
  --runtime-config configs/runtime.yaml \
  --camera-config configs/camera.yaml \
  --output /absolute/path/to/new_output \
  --stage all
```

Input contract:

- `submaps/poses.json`: consecutive node IDs, timestamps `t`, translation `p`, quaternion `q` in xyzw order.
- `submaps/NNNN.npy`: LiDAR points in each keyframe's local frame.
- `vision/data/frames/NNNN_cC.png`: indexed camera images for cameras 0–3.
- `vision/data/masks/mask2_cC.png`: rig/ceiling masks in original image coordinates.
- `vision/data/global_dinov2.npz`: normalized descriptors `[node,4,3,dimension]`, plus `nodes`; three rectified views per camera.
- `runtime.yaml`: extrinsics under `extrin_calib.cameras`; `camera.yaml`: intrinsics under `cam_0`–`cam_3`. Included files describe the paper's rig; use matching calibration for other data.

Stages: `prepare`, `verify`, `optimize`. Outputs include floor hypotheses, ground measurements, shortlists, candidate measurements, gate decisions, sequence-supported pairs, poses and summary metrics. The sequence-support query gap is 24 s, with a history-gap multiplier of 1.5. Temporal-diversity bins and spatial-anchor time windows retain their respective 12 s settings.

The frame extraction, global-descriptor and mask scripts are acquisition-specific preparation scripts. Their node ranges, bag paths and masks must match the recording. The offline entry point consumes prepared descriptors and frames.

Original reconstruction experiments used per-floor ground-measurement selection in their experiment scripts. Outputs and scripts are retained under `research/experiments`. The general entry point fits observations from the supplied submaps. Reproducing those maps requires matching sensor inputs and measurement selection.

## Expected outputs

| Output | Value |
|---|---|
| Four-floor accepted constraints | 26: F1=11, F2=3, F3=7, F4=5 |
| Single-floor accepted constraints | 17 |
| Sequence sensitivity at 24 s | 28 supported pairs, 26 accepted constraints |
| Adjacent-floor reference means | 4.201, 4.255, 4.169 m |
| Complete-method height MAE | 3.83 cm |
| NCD R@1 / R@5 | 30/31 and 31/31 with ±1-keyframe tolerance |
| NCD accepted constraints | 100, with 96 satisfying the spatial criterion |

## Packaging corrections

The public pipeline uses 24 s sequence support. Camera/submap helpers honour the paths passed by the entry point. Historical reverse-route diagnostic files are optional when importing the verifier. Experimental measurements remain unchanged. Full sensor processing and GTSAM optimization were not rerun during publication cleanup.
