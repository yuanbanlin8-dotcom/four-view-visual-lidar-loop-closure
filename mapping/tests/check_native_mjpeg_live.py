#!/usr/bin/env python3
"""Short live check for Seeker native MJPEG and its four raw counterparts."""

import statistics
import threading

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image


RAW_TOPICS = (
    "/fisheye/left/image_raw",
    "/fisheye/right/image_raw",
    "/fisheye/bright/image_raw",
    "/fisheye/bleft/image_raw",
)


class Probe:
    def __init__(self):
        self.condition = threading.Condition()
        self.compressed = []
        self.raw_by_stamp = [dict() for _ in RAW_TOPICS]
        rospy.Subscriber(
            "/fisheye/all/compressed",
            CompressedImage,
            self.compressed_callback,
            queue_size=5,
            buff_size=16 * 1024 * 1024,
        )
        for index, topic in enumerate(RAW_TOPICS):
            rospy.Subscriber(
                topic,
                Image,
                lambda message, camera=index: self.raw_callback(camera, message),
                queue_size=5,
                buff_size=8 * 1024 * 1024,
            )

    def compressed_callback(self, message):
        with self.condition:
            self.compressed.append(message)
            self.condition.notify_all()

    def raw_callback(self, camera, message):
        with self.condition:
            stamp = message.header.stamp.to_nsec()
            self.raw_by_stamp[camera][stamp] = message
            if len(self.raw_by_stamp[camera]) > 40:
                del self.raw_by_stamp[camera][min(self.raw_by_stamp[camera])]
            self.condition.notify_all()

    def wait(self, count, timeout):
        deadline = rospy.Time.now().to_sec() + timeout
        with self.condition:
            while len(self.compressed) < count and not rospy.is_shutdown():
                remaining = deadline - rospy.Time.now().to_sec()
                if remaining <= 0:
                    break
                self.condition.wait(min(remaining, 0.2))
        if len(self.compressed) < count:
            raise RuntimeError(
                f"received only {len(self.compressed)} native MJPEG frames in {timeout:.1f}s"
            )


def main():
    rospy.init_node("check_native_mjpeg_live", anonymous=True)
    probe = Probe()
    probe.wait(20, 5.0)
    messages = probe.compressed[-20:]
    stamps = [message.header.stamp.to_sec() for message in messages]
    intervals = [right - left for left, right in zip(stamps, stamps[1:])]
    payloads = [len(message.data) for message in messages]

    compressed_by_stamp = {
        message.header.stamp.to_nsec(): message for message in messages
    }
    common_stamps = set(compressed_by_stamp)
    for raw_frames in probe.raw_by_stamp:
        common_stamps.intersection_update(raw_frames)
    if not common_stamps:
        ranges = [
            (min(frames, default=None), max(frames, default=None))
            for frames in probe.raw_by_stamp
        ]
        raise RuntimeError(
            f"no timestamp common to native MJPEG and all raw cameras; raw ranges={ranges}"
        )
    matched_stamp = max(common_stamps)
    matched_compressed = compressed_by_stamp[matched_stamp]
    mosaic = cv2.imdecode(
        np.frombuffer(matched_compressed.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if mosaic is None or mosaic.shape[:2] != (5120, 1088):
        raise RuntimeError(f"unexpected decoded mosaic shape: {None if mosaic is None else mosaic.shape}")

    matching = []
    for camera in range(4):
        raw = probe.raw_by_stamp[camera][matched_stamp]
        if (raw.height, raw.width, raw.encoding) != (1280, 1088, "bgr8"):
            raise RuntimeError(
                f"camera {camera} raw format is {raw.width}x{raw.height} {raw.encoding}"
            )
        matching.append(raw)

    raw_sequences = {message.header.seq for message in matching}
    if len(raw_sequences) != 1:
        raise RuntimeError(f"restored camera sequence numbers differ: {raw_sequences}")

    average_bytes = statistics.mean(payloads)
    rate_hz = 1.0 / statistics.mean(intervals)
    bandwidth_mib = average_bytes * rate_hz / (1024 * 1024)
    print(
        f"PASS frames={len(messages)} rate={rate_hz:.2f}Hz "
        f"mjpeg={average_bytes / (1024 * 1024):.3f}MiB/frame "
        f"bandwidth={bandwidth_mib:.2f}MiB/s mosaic=1088x5120 "
        f"raw=4x1088x1280 stamp={matched_stamp} camera_sequences=matched"
    )


if __name__ == "__main__":
    main()
