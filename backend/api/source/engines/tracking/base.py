from abc import ABC, abstractmethod


class Track:
    """Tracker-agnostic track result.

    Duck-type compatible with deep_sort_realtime's Track so the controller
    can consume tracks from any tracker through the same interface.
    """

    def __init__(self, track_id, ltrb, class_id, conf=0.0, time_since_update=0):
        self.track_id = track_id
        self._ltrb = ltrb
        self._class_id = class_id
        self.conf = conf
        self.time_since_update = time_since_update

    def is_confirmed(self):
        return True

    def to_ltrb(self):
        return self._ltrb

    def get_det_class(self):
        return self._class_id


class BaseTracker(ABC):
    """Common interface every tracker implementation must provide.

    The controller pipeline is:
        detections = tracker.extract_detections(boxes)
        tracks = tracker.update_tracks(detections, frame=frame)

    where `boxes` is the VehicleDetector output: a list of
    (label, conf, [x1, y1, x2, y2]) tuples.
    """

    @abstractmethod
    def extract_detections(self, boxes):
        """Convert detector output into this tracker's input format."""

    @abstractmethod
    def update_tracks(self, detections, frame=None):
        """Advance the tracker one frame; return track objects exposing
        is_confirmed(), time_since_update, track_id, to_ltrb() and
        get_det_class()."""
