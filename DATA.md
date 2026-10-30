# Experimental data

## Dataset coverage

| Dataset | Included records | Sensor-data access |
|---|---|---|
| Single-floor indoor flight, 20261008 | Candidate pairs, pose estimates, comparison and ablation outputs, corridor profiles | ROS bags and dense point clouds are outside this public release; contact the repository maintainer for availability |
| Four-floor indoor flight, 20261013 | Candidate decisions, ground-plane measurements, floor hypotheses, trajectories, loop checks, sequence/ground/camera ablations | ROS bags and dense point clouds are outside this public release; contact the repository maintainer for availability |
| NCD Quad-Easy | 66-node adapter poses, query/positive labels, candidate ranks, accepted constraints and baseline metrics | Obtain source sensor data from the Newer College Dataset authors |

NCD uses `adapter_GT_3s`: the input node poses come from dataset ground truth. R@k uses 31 canonical queries, a 2 m spatial criterion, a temporal separation above 30 s, and a ±1-keyframe match tolerance. The 38 positive pairs include five opposite-heading pairs. The included run retrieves 30/31 at rank 1 and 31/31 at rank 5; 96 of 100 accepted constraints satisfy the spatial criterion, covering 32/38 pairs and 0/5 opposite-heading pairs. Baseline records specify whether they represent retrieval rankings or verified output. ORB-BoW is an OpenCV stand-in; BTC verified-output coverage uses its available output.

## Experimental directories

- `20261019_upload/local/four_floor_comparison_20261016`: four-floor poses, floor measurements and external-method comparisons.
- `20261019_upload/local/single_floor_ground_comparison_20261017`: corridor ground comparison.
- `20261019_upload/local/four_floor_wall_comparison_20261017` and `single_floor_comparison_20261017`: wall and structural measurements, including mixed outcomes.
- `20261019_upload/nuc/four_floor/sequence_gap`: sequence-gap sensitivity.
- `20261021_camera_count_ablation`: forward-pair verification and four-camera comparison.
- `20261021_full_single_camera`: single-camera verification outcomes.
- `20261021_gnc_sequence`: sequence/GNC ablations, including the single-floor stress case.
- `20261022_no_icp_gate_end_to_end`: LiDAR-verification ablation.
- `20261022_single_floor_full_rerun`: single-floor candidate and loop results.
- `20261022_single_floor_z_audit`: ground and upper-wall measurements.

Experiment directory and file names retain their original identifiers, including names from earlier backend variants. The current method source is `research/multifloor`. JSON provenance fields and historical scripts may contain original machine paths; use the public entry point for new inputs.

## Height measurements

The updated adjacent-floor reference means are 4.201, 4.255 and 4.169 m (F1–F2, F2–F3, F3–F4), measured at the ground-observation centre with a DLX-MD50. Instrument accuracy was supplied as ±(2 mm + D/10000). Individual repeated readings were not supplied. The CSV in `paper_assets/data/floor_height_comparison.csv` contains mapped heights, reference means, signed errors and MAE.

## File integrity

`SHA256SUMS.txt` lists release-file hashes. The seven SVG files were extracted byte-for-byte from the final main and supplementary Word documents; embedded images were preserved without resampling. The manuscript drafts, review correspondence, development archives, model weights and raw sensor releases are outside this public repository.
