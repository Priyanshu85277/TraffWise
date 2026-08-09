import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.mock_env import setup_mocks
setup_mocks()

from api.source.engines.tracking.base import BaseTracker, Track
from api.source.engines.tracking.bytetrack import ByteTrack, to_xywh
from api.source.engines.tracking.deepsort import DeepSORT
from api.source.engines.tracking.factory import create_tracker

LABELS = {"bus": 0, "car": 1, "motorbike": 2, "truck": 3}


def make_config(tracker=None):
    config = {"labels": dict(LABELS)}
    if tracker is not None:
        config["tracker"] = tracker
    return config


class FactoryTest(unittest.TestCase):
    def test_selects_bytetrack(self):
        tracker = create_tracker(make_config({"type": "bytetrack"}))
        self.assertIsInstance(tracker, ByteTrack)

    def test_selects_deepsort(self):
        tracker = create_tracker(make_config({"type": "deepsort"}))
        self.assertIsInstance(tracker, DeepSORT)

    def test_defaults_to_deepsort_without_tracker_section(self):
        tracker = create_tracker(make_config())
        self.assertIsInstance(tracker, DeepSORT)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            create_tracker(make_config({"type": "nope"}))


class ByteTrackTest(unittest.TestCase):
    def test_to_xywh_converts_corners_to_center(self):
        self.assertEqual(to_xywh([100, 50, 200, 150]), [150.0, 100.0, 100, 100])

    def test_config_params_reach_bytetracker_args(self):
        from api.source.engines.tracking import bytetrack as bt_mod
        bt_mod.BYTETracker.reset_mock()

        ByteTrack(make_config({
            "type": "bytetrack",
            "track_buffer": 45,
            "match_thresh": 0.7,
            "frame_rate": 25,
        }))

        args, kwargs = bt_mod.BYTETracker.call_args
        namespace = args[0]
        self.assertEqual(namespace.track_buffer, 45)
        self.assertEqual(namespace.match_thresh, 0.7)
        self.assertEqual(namespace.track_high_thresh, 0.25)  # default kept
        self.assertFalse(hasattr(namespace, "type"))
        self.assertFalse(hasattr(namespace, "frame_rate"))
        self.assertEqual(kwargs["frame_rate"], 25)

    def test_update_tracks_wraps_rows_into_tracks(self):
        tracker = ByteTrack(make_config({"type": "bytetrack"}))
        # rows: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        tracker.tracker = MagicMock()
        tracker.tracker.update.return_value = [
            [10.0, 20.0, 110.0, 220.0, 7.0, 0.9, 1.0, 0.0],
            [300.0, 40.0, 400.0, 140.0, 8.0, 0.6, 3.0, 1.0],
        ]

        tracks = tracker.update_tracks(MagicMock(), frame=None)

        self.assertEqual(len(tracks), 2)
        first = tracks[0]
        self.assertIsInstance(first, Track)
        self.assertTrue(first.is_confirmed())
        self.assertEqual(first.time_since_update, 0)
        self.assertEqual(first.track_id, 7)
        self.assertEqual(first.to_ltrb(), [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(first.get_det_class(), 1)
        self.assertEqual(tracks[1].track_id, 8)
        self.assertEqual(tracks[1].get_det_class(), 3)

    def test_trackers_implement_base_interface(self):
        self.assertTrue(issubclass(ByteTrack, BaseTracker))
        self.assertTrue(issubclass(DeepSORT, BaseTracker))


if __name__ == "__main__":
    unittest.main()
