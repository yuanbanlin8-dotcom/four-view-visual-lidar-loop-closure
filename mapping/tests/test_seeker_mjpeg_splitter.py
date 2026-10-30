#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "desktop" / "seeker_mjpeg_splitter.py"
SPEC = importlib.util.spec_from_file_location("seeker_mjpeg_splitter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SeekerMjpegSplitterTest(unittest.TestCase):
    def test_vertical_camera_order_and_dimensions(self):
        colors = ((15, 25, 220), (30, 210, 40), (200, 45, 35), (90, 110, 130))
        frames = [np.full((48, 64, 3), color, dtype=np.uint8) for color in colors]
        ok, encoded = cv2.imencode(".jpg", np.vstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(ok)

        decoded = MODULE.decode_mosaic(encoded.tobytes(), 4)

        self.assertEqual([frame.shape for frame in decoded], [(48, 64, 3)] * 4)
        for frame, expected in zip(decoded, colors):
            actual = frame[24, 32].astype(int)
            self.assertLess(np.max(np.abs(actual - np.array(expected))), 8)

    def test_rejects_invalid_height(self):
        image = np.zeros((101, 64, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            MODULE.decode_mosaic(encoded.tobytes(), 4)


if __name__ == "__main__":
    unittest.main()
