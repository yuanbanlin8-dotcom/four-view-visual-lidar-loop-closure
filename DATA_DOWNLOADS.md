# Indoor dataset downloads

Dataset files are attached to this repository's GitHub Releases. Download the attachments through the release pages; cloning the source repository does not download them.

## Single-floor recording: 20261008_145008

[Download release](https://github.com/yuanbanlin8-dotcom/four-view-visual-lidar-loop-closure/releases/tag/dataset-20261008-floor1)

The recording archive is split into four parts. Download all four `recording.tar.gz.part-*` files and both checksum text files into the same directory. `analysis.tar.gz` and `audits.tar.gz` contain historical analysis outputs and recording audit records. The recording includes `images_0.bag`, `sensors_0.bag`, `map_raw.pcd`, `map_5cm.pcd`, `trajectory.txt`, runtime configuration and logs.

With every release attachment downloaded, run on Linux:

```bash
sha256sum -c SHA256SUMS.txt
cat recording.tar.gz.part-00 recording.tar.gz.part-01 recording.tar.gz.part-02 recording.tar.gz.part-03 | tar -xz
(cd 20261008_145008 && sha256sum -c ../ORIGINAL_SHA256SUMS.txt)
tar -xzf analysis.tar.gz
tar -xzf audits.tar.gz
```

`VERIFIED_INVENTORY.json` lists all 17 recording files, their sizes and SHA-256 digests. Migration verified the archive attachments against the source release and all 17 decompressed recording files against `ORIGINAL_SHA256SUMS.txt`. The analysis archive contains several historical method variants; the paper's result directories are indexed in [DATA.md](DATA.md).

## Four-floor prepared inputs: 20261013_160929

[Download release](https://github.com/yuanbanlin8-dotcom/four-view-visual-lidar-loop-closure/releases/tag/dataset-20261013-four-floor-inputs)

Download `four_floor_20261013_prepared_inputs.tar.gz`, `FOUR_FLOOR_SHA256SUMS.txt` and `four_floor_input_manifest.json`. The package contains:

| Input | Coverage |
|---|---|
| Frontend poses | 243 timestamped nodes |
| LiDAR submaps | 243 NPY files |
| Camera frames | 972 PNG files: four cameras for each node |
| Masks | Four camera masks |
| DINOv2 descriptors | Array shape `[243, 4, 3, 1536]` |
| Calibration | `runtime.yaml` and `camera.yaml` |

Run from the repository root after placing the downloaded archive and checksum file there:

```bash
sha256sum -c FOUR_FLOOR_SHA256SUMS.txt
tar -xzf four_floor_20261013_prepared_inputs.tar.gz
python -m research.multifloor.run \
  --submaps "$PWD/four_floor_20261013/submaps" \
  --vision-input "$PWD/four_floor_20261013/vision/data" \
  --runtime-config "$PWD/four_floor_20261013/runtime.yaml" \
  --camera-config "$PWD/four_floor_20261013/camera.yaml" \
  --output "$PWD/outputs/four_floor_reproduction" \
  --stage all
```

Install the pipeline dependencies described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before running the last command. The entry point uses a 24 s sequence-support query gap. The prepared input package was checked for complete node/camera indexing, readable images and finite submap/descriptor values. Its pose-file SHA-256 is `ee87f177e1b509f06bbec415adf071c1ec1a9e800db3a812ecb13f0fc1f3c921`, matching the experiment input manifest. This publication check did not execute a new full perception/optimization run.

Full four-floor raw ROS bags and dense reconstructed maps are not included in this prepared-input package. The package provides inputs for the offline loop-closure backend.

## Other resources

Obtain NCD source sensor data from the Newer College Dataset authors. This repository includes NCD evaluation inputs and results as described in [DATA.md](DATA.md). Model weights are obtained from their upstream projects.
