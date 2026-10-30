#!/usr/bin/env bash
set -euo pipefail
cd /home/nuc/omni_livo_fisheye_ws
python3 tests/prepare.py
g++ -std=c++14 -O2 -I/usr/include/eigen3 -Isrc/rpg_vikit/vikit_common/include tests/test_seeker_model.cpp $(pkg-config --cflags --libs opencv4) -o tests/test_seeker_model
./tests/test_seeker_model | tee reports/model-test.txt
g++ -std=c++14 -O2 -I/usr/include/eigen3 -Isrc/rpg_vikit/vikit_common/include -Isrc/Omni-LIVO/include tests/test_wide_geometry.cpp $(pkg-config --cflags --libs opencv4) -o tests/test_wide_geometry
./tests/test_wide_geometry | tee reports/wide-geometry-test.txt
