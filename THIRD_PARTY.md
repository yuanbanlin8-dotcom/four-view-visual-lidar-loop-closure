# Third-party software

The root LICENSE contains GPL-2.0. Existing file-level notices apply to their respective files.

- **Omni-LIVO:** the mapping frontend in `mapping/src/Omni-LIVO`, with local camera, buffering, gravity-alignment and integration modifications. Its license and attribution are retained.
- **rpg_vikit:** bundled camera geometry and vision utilities in `mapping/src/rpg_vikit`; upstream notices are retained.
- **DINOv2, SuperPoint, LightGlue:** external feature models/libraries used by the visual frontend. Install from their upstream distributions under their respective terms; model weights are external downloads.
- **BTC, RTAB-Map and Scan Context:** baseline experimental outputs are included. BTC implementation source is obtained from its authors; no BTC implementation is distributed in this release.
- **ROS, PCL, OpenCV, Ceres, GTSAM, TBB, Livox ROS Driver 2:** external dependencies.

NCD Quad-Easy source sensor data are distributed by the Newer College Dataset authors under their access terms. This repository includes the derived evaluation records used in the paper.
