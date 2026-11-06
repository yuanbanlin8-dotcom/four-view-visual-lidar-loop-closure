# Four-View Visual–LiDAR Loop Closure for Drift Correction in Multi-Storey Indoor Mapping

Paper-support repository containing the algorithm, experimental records, figures and table data. The current manuscript edition is prepared for Scientific Reports and uses five main figures and three main tables.

See the [current manuscript and figure/table index](paper_assets/scientific_reports_20261106/README.md).

## Contents

| Directory | Contents |
|---|---|
| `research/multifloor/` | Four-view retrieval, visual/LiDAR verification, route-sequence support, GNC and floor factors |
| `research/vision/src/` | Camera projection, DINOv2 descriptors, SuperPoint/LightGlue and depth-supported pose estimation |
| `mapping/src/` | Omni-LIVO frontend and camera utilities |
| `configs/` | Camera intrinsics and sensor runtime calibration |
| `research/experiments/` | Single-floor and four-floor comparisons and ablations |
| `research/eval/ncd_quad_easy/` | NCD candidates, poses, evaluation labels and baseline scores |
| `paper_assets/scientific_reports_20261106/` | Current manuscript, supplementary information, five main figures and three main tables |
| `paper_assets/figures/` | Editable source SVGs with the earlier figure numbering |
| `paper_assets/tables/` | Earlier CSV tables; current Tables 1–3 and S1–S4 are in the Scientific Reports edition |
| `paper_assets/data/` | Corridor profiles, loop locations, sequence sensitivity and updated floor-height comparisons |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands, [DATA.md](DATA.md) for data coverage and [paper_assets/README.md](paper_assets/README.md) for the figure/table index.

## Dataset downloads

- [Single-floor recording, 2026-10-08](https://github.com/yuanbanlin8-dotcom/four-view-visual-lidar-loop-closure/releases/tag/dataset-20261008-floor1): camera and LiDAR/IMU bags, point clouds, trajectory and offline analysis archives.
- [Four-floor prepared inputs, 2026-10-13](https://github.com/yuanbanlin8-dotcom/four-view-visual-lidar-loop-closure/releases/tag/dataset-20261013-four-floor-inputs): 243 submaps and poses, 972 camera images, descriptors, masks and calibration.

See [DATA_DOWNLOADS.md](DATA_DOWNLOADS.md) for extraction and checksum commands. Full four-floor raw bags and dense maps remain to be added.

The sequence-support query gap is **24 s**. The visual gate retains the paper's multi-camera requirement: at least two supported cameras. The detailed thresholds are implemented in `research/multifloor/verify.py`.

## Quick checks

```bash
python -m pip install numpy scipy
python tools/check_release.py
python -m unittest research.multifloor.test_multifloor research.multifloor.tests.test_backend research.multifloor.tests.test_observation_reuse -v
```

GTSAM is needed for the backend unit checks. The full perception pipeline also uses OpenCV, PyYAML, PyTorch and LightGlue. The source environment dependency record is in `research/requirements.txt`.

## 中文说明

本仓库保留支撑论文的算法、实验结果、图表和标定配置。最终图表位于 `paper_assets`，实验原始数值位于 `research/experiments`。序列支持的 query gap 统一为 24 秒。

图中的四层路线及截图上的回环位置属于空间展示；精确回环编号和位姿以实验记录为准。相邻楼层高度比较使用重测均值 4.201、4.255、4.169 m。旧实验记录中的参考层高保留其原始值，最终比较数据见 `paper_assets/data/floor_height_comparison.csv`。

一层完整采集包（ROS bags、点云、轨迹和配置）及四层后端输入包已通过上述 Release 提供下载。四层完整原始 ROS bags、完整稠密点云和模型权重仍需另外准备。下载与校验步骤见 `DATA_DOWNLOADS.md`，运行条件见复现说明。

## License and attribution

See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md). Upstream source notices are retained.
