from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from .base import BaseTracker, Track

DEFAULTS = dict(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
    frame_rate=30,
)


class _Detections:
    """Minimal results shim BYTETracker.update() expects: numpy `xywh`,
    `conf`, `cls` attributes plus boolean-mask indexing."""

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


def to_xywh(box):
    """[x1, y1, x2, y2] -> [cx, cy, w, h]"""
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


class ByteTrack(BaseTracker):
    """ByteTrack via Ultralytics' built-in implementation.

    Motion-only (Kalman + IoU association): no per-frame ReID network,
    so the tracking step costs <1 ms and runs the same on CPU and GPU.
    """

    def __init__(self, config):
        self.config = config
        params = {**DEFAULTS, **(config.get("tracker") or {})}
        params.pop("type", None)
        frame_rate = params.pop("frame_rate")
        try:
            self.tracker = BYTETracker(
                SimpleNamespace(**params), frame_rate=frame_rate)
        except TypeError:
            # ultralytics >= 8.4 dropped the frame_rate argument;
            # track_buffer is interpreted directly as frames.
            self.tracker = BYTETracker(SimpleNamespace(**params))
        self.class_names = list(config["labels"].keys())

    def extract_detections(self, boxes):
        if len(boxes) == 0:
            return _Detections(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        xywh = np.asarray([to_xywh(box) for _, _, box in boxes],
                          dtype=np.float32)
        conf = np.asarray([conf for _, conf, _ in boxes], dtype=np.float32)
        cls = np.asarray([label for label, _, _ in boxes], dtype=np.float32)
        return _Detections(xywh, conf, cls)

    def update_tracks(self, detections, frame=None):
        # rows: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        rows = self.tracker.update(detections, img=frame)
        return [
            Track(
                track_id=int(row[4]),
                ltrb=[float(row[0]), float(row[1]),
                      float(row[2]), float(row[3])],
                class_id=int(row[6]),
                conf=float(row[5]),
            )
            for row in rows
        ]
